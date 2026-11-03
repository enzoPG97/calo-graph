import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch  # noqa: E402

from calograph import generate_pair_event  # noqa: E402
from calograph.graph import build_graph, to_pyg  # noqa: E402
from calograph.model import EdgeClassifier  # noqa: E402


def _data(sep=2.0, seed=0):
    # low energies keep the graphs small and the tests fast
    event = generate_pair_event(sep, energies_gev=(3.0, 3.0), seed=seed)
    return to_pyg(build_graph(event, k=6))


def _model(**kwargs):
    torch.manual_seed(0)
    return EdgeClassifier(**kwargs)


def test_output_has_one_finite_logit_per_edge():
    data = _data()
    out = _model()(data)
    assert out.shape == (data.edge_index.shape[1],)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_reverse_edges_get_the_same_logit():
    data = _data()
    out = _model().eval()(data)
    src, dst = data.edge_index
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(src, dst))}
    reverse = torch.tensor([lookup[(int(j), int(i))] for i, j in zip(src, dst)])
    assert torch.allclose(out, out[reverse], atol=1e-5)


def test_prediction_does_not_depend_on_hit_ordering():
    data = _data()
    model = _model().eval()
    out = model(data)

    perm = torch.randperm(data.num_nodes, generator=torch.Generator().manual_seed(1))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    shuffled = data.clone()
    shuffled.x = data.x[perm]                       # new node i is old node perm[i]
    shuffled.edge_index = inverse[data.edge_index]  # relabel the edge end points
    assert torch.allclose(model(shuffled), out, atol=1e-4)


def test_batch_of_graphs_gives_same_logits_as_single_graphs():
    model = _model().eval()
    a, b = _data(seed=1), _data(seed=2)
    with torch.no_grad():
        batched = model(Batch.from_data_list([a, b]))
        single = torch.cat([model(a), model(b)])
    assert torch.allclose(batched, single, atol=1e-4)


def test_gradients_reach_every_parameter():
    data = _data()
    model = _model()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(data), data.edge_label)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing


def test_loss_decreases_when_fitting_one_event():
    data = _data(sep=4.0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def loss_fn():
        return torch.nn.functional.binary_cross_entropy_with_logits(model(data), data.edge_label)

    first = loss_fn().item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
    assert loss_fn().item() < 0.7 * first


def test_parameter_count_is_small():
    n = sum(p.numel() for p in _model().parameters())
    assert 1_000 < n < 100_000

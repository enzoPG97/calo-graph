# calo-graph
calo-graph: separating overlapping showers with a GNN using truth labels. 
_Project idea: development of a Toy caloerimtric GNN for shower separation._
  ![CI](https://github.com/enzoPG97/calo-graph/actions/workflows/ci.yml/badge.svg)

Toy project exploring graph neural networks for separating overlapping
particle showers in a layered calorimeter, using simulation truth for labels.

**Scope.** Events are generated from a simple shower model, with each hit
keeping the ID of its truth particle. Hits are connected in a graph, and a GNN
is trained to decide whether two hits come from the same particle (edge
classification), from which clusters are built. The focus is on merged
topologies (two close showers), as a small-scale analogue of truth-based
clustering and association in calorimeter reconstruction (e.g. HGCAL/TICL).
This is a learning project on **simulated toy data**, not a realistic detector
simulation, and it runs on CPU only.

## Status
Work in progress. See the repository issues and pull requests for progress.

## Installation
(to be completed)

## What's up
* Much faster + more organized implementations of logistic regression and preconditioning net. Yay lightning
* I actually know how to scale and test models in PyTorch now! Yay Joey
* Model currently training across subjects... but do I have the right approach?

## Questions 
* How best to evaluate/compare models?
* Inefficient to re-train the whole model when a huge amount of the data is overlapping... 
    - checkpoint and train on the holdout subject (but not session)?
    - how to prevent this from overfitting?

## Triage (what goes in the thesis)
* Fig 1: overall methods (boring stuff like memory task, scalp caps, etc.)
* Fig 2: network architecture
* Fig 3: look it trains, e.g. sklearn (?)
* Fig 4: it might be better with more data
* Fig 5: how preconditioning across sessions compares
* Fig 6: how preconditioning across subjects compares
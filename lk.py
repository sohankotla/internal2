import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

# Define the names of the variables
names = ["age", "chol", "target"]

# Create the data dictionary
data = {
    "age": [17, 22, 22, 34],
    "chol": [0, 190, 1, 12],
    "target": [1, 0, 1, 0]  # Changed from ';' to ':'
}

# Create a DataFrame
heart_disease = pd.DataFrame(data)

# Define the structure of the Bayesian Network
model = BayesianNetwork([('age', 'target'), ('chol', 'target')])  # Example structure

# Fit the model using Maximum Likelihood Estimator
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Perform inference using Variable Elimination
vas = VariableElimination(model)

# Define the query
query = vas.query(variables=['target'], evidence={'age': 46})

# Print the query result
print(query)
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

def load_csv(dataset_file, tasks, featurizer='ECFP', split='random'):

  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = dc.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = dc.feat.RawFeaturizer()
  elif featurizer == 'AdjacencyConv':
    featurizer = dc.feat.AdjacencyFingerprint(
        max_n_atoms=150, max_valence=6)

  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  all_dataset = (train, valid, test)
  transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]

  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)


  return tasks, all_dataset, transformers

tasks, datasets, transformers = load_csv('SMILES&Energy.csv', ['energy'], featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

# Batch size of models
batch_size = 64

model = dc.models.GraphConvTensorGraph(
    len(tasks), batch_size=batch_size, learning_rate=0.001, mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=5000)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

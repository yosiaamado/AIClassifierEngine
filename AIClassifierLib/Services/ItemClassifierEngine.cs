using AIClassifierLib.Interface;
using AIClassifierLib.Models;
using Microsoft.ML;
using NHunspell;
using System.Data;

namespace AIClassfierLib.Services
{
    internal class ItemClassifierEngine : IItemClassifierEngine, IDisposable
    {
        private MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<ItemData, ItemPrediction> _predictor;
        private Hunspell _spellChecker;
        private readonly SemaphoreSlim _initLock = new(1, 1);
        public bool IsInitialized => _initialized;
        private bool _initialized = false;
        private void EnsureInitialized()
        {
            if (!_initialized || _predictor == null)
                throw new InvalidOperationException("ItemClassifierEngine is not initialized. Call InitAsync() first.");
        }
        /// <summary>
        /// Initializes the classifier engine by loading or training the model.
        /// This must be called once before using any prediction or correction methods.
        /// </summary>
        public async Task InitAsync(IEnumerable<ItemData> dataset, string modelPath, string affPath = null, string dicPath = null)
        {
            await _initLock.WaitAsync();
            try
            {
                if (_initialized) return;

                _mlContext = new MLContext();

                if (File.Exists(modelPath))
                {
                    using var stream = File.OpenRead(modelPath);
                    LoadModel(stream);
                }
                else
                {
                    var data = _mlContext.Data.LoadFromEnumerable(dataset);
                    Train(data);
                    using var stream = File.Create(modelPath);
                    SaveModel(stream);
                }

                if(affPath is not null || dicPath is not null)
                {
                    _spellChecker = new Hunspell(affPath, dicPath);
                }

                _initialized = true;
            }
            finally
            {
                _initLock.Release();
            }
        }
        public string Predict(string name)
        {
            EnsureInitialized();
            return _predictor.Predict(new ItemData { Name = name }).PredictedCategory;
        }
        public void Retrain(IEnumerable<ItemData> items)
        {
            EnsureInitialized();
            var data = _mlContext.Data.LoadFromEnumerable(items);
            Train(data);
        }
        private void Train(IDataView data)
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Category")
                .Append(_mlContext.Transforms.Text.FeaturizeText("Features", nameof(ItemData.Name)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Category", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedCategory"));

            _model = pipeline.Fit(data);
            _predictor = _mlContext.Model.CreatePredictionEngine<ItemData, ItemPrediction>(_model);
        }

        private void SaveModel(Stream output)
        {
            EnsureInitialized();
            _mlContext.Model.Save(_model, inputSchema: null, stream: output);
        }

        private void LoadModel(Stream modelStream)
        {
            EnsureInitialized();
            _model = _mlContext.Model.Load(modelStream, out _);
            _predictor = _mlContext.Model.CreatePredictionEngine<ItemData, ItemPrediction>(_model);
        }

        public string KBBISpelling(string word)
        {
            EnsureInitialized();
            if (_spellChecker == null) return word;

            var suggestion = _spellChecker.Suggest(word);
            return suggestion.FirstOrDefault() ?? word;
        }
        public string AutoCorrect(string input, IEnumerable<string> knownWords)
        {
            EnsureInitialized();
            int Levenshtein(string a, string b)
            {
                int[,] dp = new int[a.Length + 1, b.Length + 1];
                for (int i = 0; i <= a.Length; i++) dp[i, 0] = i;
                for (int j = 0; j <= b.Length; j++) dp[0, j] = j;

                for (int i = 1; i <= a.Length; i++)
                {
                    for (int j = 1; j <= b.Length; j++)
                    {
                        int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                        dp[i, j] = new[] {
                            dp[i - 1, j] + 1,
                            dp[i, j - 1] + 1,
                            dp[i - 1, j - 1] + cost
                        }.Min();
                    }
                }

                return dp[a.Length, b.Length];
            }

            var bestMatch = knownWords
                .OrderBy(word => Levenshtein(input, word))
                .FirstOrDefault();

            return bestMatch ?? input;
        }

        public void Dispose()
        {
            _spellChecker?.Dispose();
        }
    }
}

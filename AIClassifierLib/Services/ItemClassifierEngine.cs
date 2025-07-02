using AIClassifierLib.Interface;
using AIClassifierLib.Models;
using Microsoft.ML;
using NHunspell;
using System.Data;

namespace AIClassfierLib.Services
{
    internal class ItemClassifierEngine : IItemClassifierEngine, IDisposable
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<ItemData, ItemPrediction> _predictor;
        private readonly Hunspell _spellChecker;

        public ItemClassifierEngine(IEnumerable<ItemData> items, string? affPath = null, string? dicPath = null)
        {
            _mlContext = new MLContext();
            _spellChecker = new Hunspell(affPath, dicPath);

            var data = _mlContext.Data.LoadFromEnumerable(items);
            Train(data);
        }
        public string Predict(string name)
        {
            return _predictor.Predict(new ItemData { Name = name }).PredictedCategory;
        }
        public void Retrain(IEnumerable<ItemData> items)
        {
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

        public void SaveModel(Stream output)
        {
            _mlContext.Model.Save(_model, inputSchema: null, stream: output);
        }

        public void LoadModel(Stream modelStream)
        {
            _model = _mlContext.Model.Load(modelStream, out _);
            _predictor = _mlContext.Model.CreatePredictionEngine<ItemData, ItemPrediction>(_model);
        }

        public string KBBISpelling(string word)
        {
            var suggestion = _spellChecker.Suggest(word);
            return suggestion.FirstOrDefault() ?? word;
        }
        public string AutoCorrect(string input, IEnumerable<string> knownWords)
        {
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

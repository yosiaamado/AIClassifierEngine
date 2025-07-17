using AIClassifierLib.Interface;
using AIClassifierLib.Models;
using Microsoft.ML;
using NHunspell;

namespace AIClassifierLib.Services
{
    internal class ItemSuggestionEngine : IItemSuggestionEngine
    {
        private MLContext _mlContext;
        private ITransformer _model;
        private readonly SemaphoreSlim _initLock = new(1, 1);
        private PredictionEngine<TransactionHistory, SuggestionPrediction> _predictor;
        public bool IsInitialized => _initialized;
        private bool _initialized = false;

        private void EnsureInitialized()
        {
            if (!_initialized || _predictor == null)
                throw new InvalidOperationException("ItemSuggestionEngine is not initialized. Call InitAsync() first.");
        }
        public async Task InitAsync(IEnumerable<TransactionHistory> dataset, string modelPath)
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
                    TrainModel(dataset);
                    using var stream = File.Create(modelPath);
                    SaveModel(stream);
                }

                _initialized = true;
            }
            finally
            {
                _initLock.Release();
            }
        }

        private void TrainModel(IEnumerable<TransactionHistory> data)
        {
            var trainingData = _mlContext.Data.LoadFromEnumerable(data);

            var pipeline = _mlContext.Transforms.Conversion
                .MapValueToKey("Label", nameof(TransactionHistory.Item))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("UserId"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("DayOfWeek"))
                .Append(_mlContext.Transforms.Concatenate("Features", "UserId", "Hour", "DayOfWeek"))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _model = pipeline.Fit(trainingData);
            _predictor = _mlContext.Model.CreatePredictionEngine<TransactionHistory, SuggestionPrediction>(_model);
        }
        public void Retrain(IEnumerable<TransactionHistory> dataset)
        {
            EnsureInitialized();
            TrainModel(dataset);
        }

        private void SaveModel(Stream stream)
        {
            var schema = _mlContext.Data.LoadFromEnumerable(new List<TransactionHistory>()).Schema;
            _mlContext.Model.Save(_model, schema, stream);
        }

        private void LoadModel(Stream stream)
        {
            var model = _mlContext.Model.Load(stream, out var schema);
            _predictor = _mlContext.Model.CreatePredictionEngine<TransactionHistory, SuggestionPrediction>(model);
        }

        public string Suggest(string userId, int hour, string dayOfWeek)
        {
            var input = new TransactionHistory
            {
                UserId = userId,
                Hour = hour,
                DayOfWeek = dayOfWeek
            };

            var prediction = _predictor.Predict(input);
            return prediction.SuggestedItem;
        }
    }
}

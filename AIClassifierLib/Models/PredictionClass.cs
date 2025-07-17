using Microsoft.ML.Data;

namespace AIClassifierLib.Models
{
    public class PredictionClass
    {
        [ColumnName("PredictedCategory")]
        public string PredictedCategory { get; set; }
    }

    public class SuggestionPrediction
    {
        [ColumnName("PredictedItem")]
        public string SuggestedItem { get; set; }
    }

}

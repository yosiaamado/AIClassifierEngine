using Microsoft.ML.Data;

namespace AIClassifierLib.Models
{
    public class ItemPrediction
    {
        [ColumnName("PredictedCategory")]
        public string PredictedCategory { get; set; }
    }

}

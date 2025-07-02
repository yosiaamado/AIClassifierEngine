using AIClassifierLib.Models;

namespace AIClassifierLib.Interface
{
    public interface IItemClassifierEngine
    {
        /// <summary>
        /// Indicates whether the engine has been initialized and is ready to use.
        /// </summary>
        bool IsInitialized { get; }
        /// <summary>
        /// Initializing engine needs for the first time used
        /// </summary>
        Task InitAsync(IEnumerable<ItemData> dataset, string modelPath, string affPath = null, string dicPath = null);
        string Predict(string input);
        void Retrain(IEnumerable<ItemData> dataset);
        string KBBISpelling(string word);
        string AutoCorrect(string input, IEnumerable<string> knownWords);
    }
}

using AIClassifierLib.Models;

namespace AIClassifierLib.Interface
{
    public interface IItemClassifierEngine
    {
        string Predict(string input);
        void Retrain(IEnumerable<ItemData> dataset);
        void LoadModel(Stream modelStream);
        void SaveModel(Stream output);
        string KBBISpelling(string word);
        string AutoCorrect(string input, IEnumerable<string> knownWords);
    }
}

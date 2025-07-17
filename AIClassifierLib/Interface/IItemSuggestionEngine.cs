using AIClassifierLib.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIClassifierLib.Interface
{
    public interface IItemSuggestionEngine
    {
        /// <summary>
        /// Indicates whether the engine has been initialized and is ready to use.
        /// </summary>
        bool IsInitialized { get; }
        /// <summary>
        /// Initializing engine needs for the first time used
        /// </summary>
        Task InitAsync(IEnumerable<TransactionHistory> dataset, string modelPath);
        string Suggest(string userId, int hour, string dayOfWeek);
        void Retrain(IEnumerable<TransactionHistory> dataset);
    }
}

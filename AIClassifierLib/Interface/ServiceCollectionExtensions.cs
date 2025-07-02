using AIClassfierLib.Services;
using AIClassifierLib.Interface;
using AIClassifierLib.Models;
using Microsoft.Extensions.DependencyInjection;

namespace AIClassifierLib.Extensions
{
    public static class ServiceInitiatior
    {
        public static IServiceCollection AddItemClassifier(this IServiceCollection services)
        {
            services.AddSingleton<IItemClassifierEngine, ItemClassifierEngine>();
            return services;
        }
        /// <summary>
        /// Use for Console App
        /// </summary>
        public static async Task<IItemClassifierEngine> CreateAsync(IEnumerable<ItemData> data, string modelPath, string affPath = null, string dicPath = null)
        {
            var engine = new ItemClassifierEngine();
            await engine.InitAsync(data, modelPath, affPath, dicPath);
            return engine;
        }
    }
}

using AIClassifierLib.Extensions;
using AIClassifierLib.Interface;
using AIClassifierLib.Models;
class Program
{
    static string projectRoot = Directory.GetParent(Directory.GetCurrentDirectory())?.Parent?.Parent?.FullName!;
    static string datasetPath = Path.Combine(projectRoot, "dataset.tsv");
    static string modelPath = Path.Combine(projectRoot, "trained-model.zip");
    static string indAff = Path.Combine(projectRoot, "dict", "Indonesia.aff");
    static string indDic = Path.Combine(projectRoot, "dict", "Indonesia.dic");
    private readonly IItemClassifierEngine _engine;
    
    public Program(IItemClassifierEngine engine)
    {
        _engine = engine;
    }
    static async Task Main(string[] args)
    {
        var dataset = LoadDataset();
        var engine = await ServiceInitiatior.CreateAsync(dataset, modelPath, indAff, indDic);

        var app = new Program(engine);
        app.Run();
    }

    public void Run()
    {
        Console.WriteLine("Klasifikasi Barang. Ketik 'exit' untuk keluar.");

        var knownWords = LoadDataset().Select(d => d.Name).Distinct().ToList();

        while (true)
        {
            Console.Write("> Nama Barang : ");
            var input = Console.ReadLine()?.Trim().ToLower();
            if (input == "exit") break;

            var corrected = _engine.AutoCorrect(input, knownWords);
            if (corrected != input)
            {
                Console.WriteLine($"Apakah yang dimaksud {corrected}?");
                Console.Write("Benar? (y/n): ");
                if (Console.ReadLine()?.Trim().ToLower() == "y")
                    input = corrected;
            }

            var result = _engine.Predict(input);
            Console.WriteLine($"Prediksi Kategori: {result}");

            Console.Write("Benar? (y/n): ");
            if (Console.ReadLine()?.Trim().ToLower() == "y")
                continue;

            var existingLabels = LoadDataset().Select(d => d.Category).Distinct().ToList();
            for (int i = 0; i < existingLabels.Count; i++)
                Console.WriteLine($"{i + 1}. {existingLabels[i]}");
            Console.WriteLine($"{existingLabels.Count + 1}. Tambah kategori baru");

            Console.Write("Input nomor atau kategori baru: ");
            var choice = Console.ReadLine();
            string newLabel;
            if (int.TryParse(choice, out int index) && index >= 1 && index <= existingLabels.Count)
                newLabel = existingLabels[index - 1];
            else if (index == existingLabels.Count + 1)
            {
                Console.Write("Masukkan nama kategori baru: ");
                newLabel = Console.ReadLine();
            }
            else
            {
                Console.WriteLine("Input tidak valid, data tidak disimpan.");
                return;
            }

            File.AppendAllText(datasetPath, $"{input}\t{newLabel}\n");
            Console.WriteLine("Data ditambahkan! Model akan retrain...");

            var newData = LoadDataset();
            _engine.Retrain(newData);
        }
    }

    private static List<ItemData> LoadDataset()
    {
        return File.ReadAllLines(datasetPath)
            .Skip(1)
            .Select(line =>
            {
                var parts = line.Split('\t');
                return new ItemData { Name = parts[0].Trim(), Category = parts[1].Trim() };
            }).ToList();
    }
}
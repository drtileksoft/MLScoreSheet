using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLScoreSheet.Core.Tests
{
    public sealed class OverlayExpectations
    {
        [JsonPropertyName("winnerMap")]
        public int[] WinnerMap { get; set; } = Array.Empty<int>();

        [JsonPropertyName("rowSums")]
        public int[][] RowSums { get; set; } = Array.Empty<int[]>();

        [JsonPropertyName("columnSums")]
        public int[][] ColumnSums { get; set; } = Array.Empty<int[]>();

        [JsonPropertyName("tableTotals")]
        public int[] TableTotals { get; set; } = Array.Empty<int>();
    }

    // --- Serializer/Converter helpers for result.Details → OverlayExpectations JSON ---
    public static class OverlayExpectationsIo
    {
        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            WriteIndented = false,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        /// <summary>
        /// Serializes the provided details object into the OverlayExpectations JSON shape.
        /// Uses reflection to avoid taking a hard dependency on the concrete Details type.
        /// Expected properties on details:
        ///   - bool[] WinnerMap
        ///   - int[,] RowSums
        ///   - int[,] ColumnSums
        ///   - int[] TableTotals
        /// </summary>
        public static void SaveDetailsSnapshot(object details, string filePath)
        {
            var snapshot = Capture(details);
            var json = JsonSerializer.Serialize(snapshot, JsonOptions);

            var path = Path.Combine(AppContext.BaseDirectory, "Assets", filePath);

            // Write only if content changed (keeps git diffs clean)
            if (File.Exists(path))
            {
                var existing = File.ReadAllText(path);
                if (existing == json) return;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, json);
        }

        private static OverlayExpectations Capture(object details)
        {
            // Pull properties via reflection
            var t = details.GetType();

            var winnerMapBool = GetRequired<bool[]>(t, details, "WinnerMap");
            var rowSums2D = GetRequired<int[,]>(t, details, "RowSums");
            var colSums2D = GetRequired<int[,]>(t, details, "ColumnSums");
            var tableTotals = GetRequired<int[]>(t, details, "TableTotals");

            // Convert to JSON-friendly shape
            return new OverlayExpectations
            {
                WinnerMap = To01(winnerMapBool),
                RowSums = ToJagged(rowSums2D),
                ColumnSums = ToJagged(colSums2D),
                TableTotals = tableTotals.ToArray()
            };
        }

        private static T GetRequired<T>(System.Type t, object instance, string name)
        {
            var p = t.GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
            if (p == null)
                throw new MissingMemberException(t.FullName, name);
            var value = p.GetValue(instance);
            if (value is null)
                throw new InvalidDataException($"Property '{name}' was null on {t.FullName}.");
            if (value is not T typed)
                throw new InvalidCastException($"Property '{name}' on {t.FullName} was '{value.GetType().FullName}', expected '{typeof(T).FullName}'.");
            return typed;
        }

        private static int[] To01(bool[] source) =>
            source.Select(b => b ? 1 : 0).ToArray();

        private static int[][] ToJagged(int[,] src)
        {
            var rows = src.GetLength(0);
            var cols = src.GetLength(1);
            var result = new int[rows][];
            for (int r = 0; r < rows; r++)
            {
                var row = new int[cols];
                for (int c = 0; c < cols; c++)
                    row[c] = src[r, c];
                result[r] = row;
            }
            return result;
        }
    }
}

using System.Threading;
using System.Threading.Tasks;

namespace MLScoreSheetCounter.Services;

public interface IGallerySaver
{
    Task SaveImageAsync(string filePath, string fileName, CancellationToken cancellationToken = default);
}

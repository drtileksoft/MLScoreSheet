using System.IO;

namespace MLScoreSheet.Core;

public interface IResourceProvider
{
    Task<Stream> OpenReadAsync(string logicalName);
}

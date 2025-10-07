using System;
using Microsoft.Extensions.DependencyInjection;

namespace MLScoreSheetCounter;

public static class ServiceHelper
{
    private static IServiceProvider? _services;

    public static void Initialize(IServiceProvider services)
    {
        _services = services;
    }

    public static T GetRequiredService<T>() where T : notnull
    {
        if (_services == null)
        {
            throw new InvalidOperationException("Služby nejsou inicializovány.");
        }

        return _services.GetRequiredService<T>();
    }
}

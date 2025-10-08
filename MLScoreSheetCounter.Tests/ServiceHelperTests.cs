using System;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace MLScoreSheetCounter.Tests;

public class ServiceHelperTests : IDisposable
{
    public ServiceHelperTests()
    {
        Reset();
    }

    public void Dispose()
    {
        Reset();
    }

    [Fact]
    public void GetRequiredService_ThrowsWhenNotInitialized()
    {
        Assert.Throws<InvalidOperationException>(() => ServiceHelper.GetRequiredService<object>());
    }

    [Fact]
    public void GetRequiredService_ReturnsRegisteredService()
    {
        var services = new ServiceCollection();
        services.AddSingleton<string>("hello");
        var provider = services.BuildServiceProvider();

        ServiceHelper.Initialize(provider);

        var result = ServiceHelper.GetRequiredService<string>();

        Assert.Equal("hello", result);
    }

    private static void Reset()
    {
        var field = typeof(ServiceHelper).GetField("_services", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
        field?.SetValue(null, null);
    }
}

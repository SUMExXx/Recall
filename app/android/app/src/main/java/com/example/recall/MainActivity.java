package com.example.recall;

import android.content.Intent;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;

import androidx.annotation.NonNull;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.MethodChannel;

public class MainActivity extends FlutterActivity {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private GenieXRunner genie;

    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        super.configureFlutterEngine(flutterEngine);
        genie = new GenieXRunner(getApplicationContext());
        new MethodChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), "geniex")
                .setMethodCallHandler((call, result) -> {
                    switch (call.method) {
                        case "generate": {
                            final String prompt = call.argument("prompt") != null
                                    ? call.argument("prompt") : "";
                            run(result, () -> genie.generate(prompt));
                            break;
                        }
                        case "isModelReady":
                            run(result, () -> genie.isModelReady());
                            break;
                        case "listModels":
                            run(result, () -> genie.listModels());
                            break;
                        case "downloadModel": {
                            final String model = call.argument("model");
                            run(result, () -> {
                                genie.downloadModel(model);
                                return null;
                            });
                            break;
                        }
                        case "registerLocalModel": {
                            final String path = call.argument("path");
                            run(result, () -> {
                                genie.registerLocalModel(path);
                                return null;
                            });
                            break;
                        }
                        case "hasStorageAccess":
                            // All-files access needed to read GenieX bundles under /sdcard.
                            result.success(Environment.isExternalStorageManager());
                            break;
                        case "requestStorageAccess": {
                            Intent i = new Intent(
                                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                                    Uri.parse("package:" + getPackageName()));
                            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                            startActivity(i);
                            result.success(null);
                            break;
                        }
                        default:
                            result.notImplemented();
                    }
                });
    }

    /** Runs blocking GenieX work off the platform thread; replies on the main thread. */
    private void run(MethodChannel.Result result, java.util.concurrent.Callable<Object> work) {
        executor.execute(() -> {
            try {
                Object value = work.call();
                mainHandler.post(() -> result.success(value));
            } catch (ModelUnavailableException e) {
                mainHandler.post(() -> result.error("GENIE_UNAVAILABLE", e.getMessage(), null));
            } catch (Exception e) {
                mainHandler.post(() -> result.error("GENIE_ERROR", e.getMessage(), null));
            }
        });
    }

    @Override
    protected void onDestroy() {
        executor.shutdown();
        if (genie != null) genie.close();
        super.onDestroy();
    }
}

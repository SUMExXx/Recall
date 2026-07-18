plugins {
    id("com.android.application")
    id("kotlin-android") // GenieX SDK is a Kotlin coroutine API; runner is Kotlin
    // The Flutter Gradle Plugin must be applied after the Android/Kotlin plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.recall"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = "27.0.12077973" // required by sherpa_onnx / sqflite / permission_handler

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    defaultConfig {
        applicationId = "com.example.recall"
        minSdk = 31 // required by the GenieX SDK (com.qualcomm.qti:geniex-android)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        // ponytail: arm64 only (target phone + GenieX ship arm64 only). Add other
        // ABIs for older devices/emulators (GenieX won't run there anyway).
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // Both sherpa_onnx (ORT 1.27) and the onnxruntime package (ORT 1.15) ship
    // libonnxruntime.so. We must keep sherpa's 1.27 (its C API is backward-
    // compatible, so it also serves the onnxruntime Dart binding, which only
    // needs 1.15's API). pickFirst alone kept the wrong (1.15) copy, so we drop
    // the onnxruntime package's copy at its source (below) and keep pickFirst as
    // a safety net.
    packaging {
        jniLibs {
            pickFirsts += "**/libonnxruntime.so"
            useLegacyPackaging = true // GenieX native libs load from extracted files
        }
    }
}

dependencies {
    implementation("com.qualcomm.qti:geniex-android:0.3.5") // on-NPU LLM runtime
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}

// Strip the onnxruntime package's older libonnxruntime.so so only sherpa's 1.27
// reaches the native-libs merge.
project(":onnxruntime").plugins.withId("com.android.library") {
    project(":onnxruntime").extensions.configure<com.android.build.gradle.LibraryExtension>("android") {
        packaging.jniLibs.excludes.add("**/libonnxruntime.so")
    }
}

flutter {
    source = "../.."
}

package ai.ainamaura.checkers

import android.Manifest
import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.darkColorScheme
import androidx.compose.ui.Modifier
import ai.ainamaura.checkers.theme.Background

class MainActivity : ComponentActivity() {

    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Fix 2: read isFirstBoot from SharedPreferences — only true on genuine first launch
        val prefs = getSharedPreferences("ainamaura_prefs", Context.MODE_PRIVATE)
        val isFirstBoot = !prefs.getBoolean("boot_complete", false)

        requestPermissionsLauncher.launch(
            arrayOf(
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.READ_MEDIA_IMAGES
            )
        )

        // Fix 3: initialize the C++ native engine on startup
        NativeEngine.initEngine()

        setContent {
            MaterialTheme(colorScheme = darkColorScheme(background = Background, surface = Background)) {
                Surface(color = Background) {
                    // Fix 1: route through real Navigation, not the inline stub
                    MainNavigation(
                        isFirstBoot = isFirstBoot,
                        onFirstBootComplete = {
                            prefs.edit().putBoolean("boot_complete", true).apply()
                        }
                    )
                }
            }
        }
    }
}

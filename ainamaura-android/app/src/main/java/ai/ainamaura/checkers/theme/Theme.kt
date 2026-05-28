package ai.ainamaura.checkers.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.graphics.Color

private val AinamauraColorScheme = darkColorScheme(
    primary = NeonBlue,
    secondary = NeonPurple,
    tertiary = NeonOrange,
    background = Background,
    surface = Background,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onTertiary = Color.White,
    onBackground = NeonTeal,
    onSurface = NeonTeal
)

@Composable
fun AinamauraCheckersTheme(
  content: @Composable () -> Unit,
) {
  MaterialTheme(colorScheme = AinamauraColorScheme, typography = Typography, content = content)
}

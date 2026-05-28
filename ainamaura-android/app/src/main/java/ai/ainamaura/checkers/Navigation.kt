package ai.ainamaura.checkers

import android.app.Application
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.ui.NavDisplay
import ai.ainamaura.checkers.ui.main.MainScreen
import ai.ainamaura.checkers.ui.main.FirstBootScreen
import ai.ainamaura.checkers.ui.main.MainScreenViewModel
import ai.ainamaura.checkers.ui.main.MainScreenViewModelFactory

@Composable
fun MainNavigation(isFirstBoot: Boolean, onFirstBootComplete: () -> Unit) {
    val backStack = rememberNavBackStack(if (isFirstBoot) FirstBoot else Main)
    val application = LocalContext.current.applicationContext as Application
    // Share the same ViewModel instance across Navigation and FirstBootScreen
    val viewModel: MainScreenViewModel = viewModel(factory = MainScreenViewModelFactory(application))

    NavDisplay(
        backStack = backStack,
        onBack = { backStack.removeLastOrNull() },
        entryProvider = entryProvider {
            entry<FirstBoot> {
                FirstBootScreen(
                    onComplete = { uri ->
                        // Feed seed image into the neural system
                        uri?.let { viewModel.handleImageSeed(it) }
                        onFirstBootComplete()
                        backStack.clear()
                        backStack.add(Main)
                    },
                    onStartVoiceCalibration = { viewModel.startListening() },
                    onStopVoiceCalibration = { viewModel.stopListening() }
                )
            }
            entry<Main> {
                MainScreen(
                    onItemClick = { navKey -> backStack.add(navKey) },
                    modifier = Modifier.safeDrawingPadding().padding(16.dp),
                    sharedViewModel = viewModel
                )
            }
        }
    )
}

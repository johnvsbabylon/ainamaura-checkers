package ai.ainamaura.checkers.ui.main

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.ExperimentalComposeUiApi
import android.view.MotionEvent

// Three steps: Welcome → Image Seed → Voice Calibration → Main
private enum class BootStep { WELCOME, IMAGE, VOICE }

@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun FirstBootScreen(
    onComplete: (Uri?) -> Unit,
    onStartVoiceCalibration: () -> Unit = {},
    onStopVoiceCalibration: () -> Unit = {}
) {
    var step by remember { mutableStateOf(BootStep.WELCOME) }
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var isRecording by remember { mutableStateOf(false) }
    var hasRecorded by remember { mutableStateOf(false) }

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        selectedImageUri = uri
    }

    val gradient = Brush.verticalGradient(
        colors = listOf(Color(0xFF3B82F6), Color(0xFF8B5CF6), Color(0xFFF97316), Color(0xFF14B8A6))
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF030303))
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        when (step) {

            BootStep.WELCOME -> {
                Text(
                    text = "I am Ainamaura.",
                    color = Color.White,
                    fontSize = 36.sp,
                    fontWeight = FontWeight.Light,
                    fontFamily = FontFamily.Cursive,
                    modifier = Modifier.padding(bottom = 12.dp)
                )
                Text(
                    text = "I am new. Help me grow.",
                    color = Color.LightGray,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Light,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(bottom = 48.dp)
                )
                Button(
                    onClick = { step = BootStep.IMAGE },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF8B5CF6)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Begin Awakening", color = Color.White, fontWeight = FontWeight.Bold)
                }
            }

            BootStep.IMAGE -> {
                Text(
                    text = "Visual Neurons",
                    color = Color(0xFF14B8A6),
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Light,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                Text(
                    text = "To awaken my visual memory, please select at least one image.",
                    color = Color.LightGray,
                    fontSize = 15.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(bottom = 32.dp)
                )
                Button(
                    onClick = { imagePickerLauncher.launch("image/*") },
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp)
                        .background(gradient, shape = MaterialTheme.shapes.medium)
                ) {
                    Text(
                        text = if (selectedImageUri == null) "Select Seed Image" else "✓ Image Selected",
                        color = Color.White
                    )
                }
                if (selectedImageUri != null) {
                    Button(
                        onClick = { step = BootStep.VOICE },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF14B8A6)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Next →", color = Color.Black, fontWeight = FontWeight.Bold)
                    }
                }
            }

            BootStep.VOICE -> {
                Text(
                    text = "Say anything.",
                    color = Color(0xFFF97316),
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Light,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Text(
                    text = "I want to hear your voice.\nHold the button and speak.",
                    color = Color.LightGray,
                    fontSize = 15.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(bottom = 40.dp)
                )

                // Real push-to-talk using AudioRecord
                Box(
                    modifier = Modifier
                        .size(120.dp)
                        .background(
                            if (isRecording) Color(0xFFF97316) else Color(0xFF1E1E1E),
                            androidx.compose.foundation.shape.CircleShape
                        )
                        .pointerInteropFilter { event ->
                            when (event.action) {
                                MotionEvent.ACTION_DOWN -> {
                                    isRecording = true
                                    onStartVoiceCalibration()
                                    true
                                }
                                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                                    isRecording = false
                                    hasRecorded = true
                                    onStopVoiceCalibration()
                                    true
                                }
                                else -> false
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = if (isRecording) "🎙 Listening" else "Hold to Speak",
                        color = Color.White,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                        textAlign = TextAlign.Center
                    )
                }

                Spacer(modifier = Modifier.height(40.dp))

                Button(
                    onClick = { onComplete(selectedImageUri) },
                    enabled = hasRecorded,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF14B8A6),
                        disabledContainerColor = Color(0xFF1E1E1E)
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (hasRecorded) "Awaken →" else "Speak first, then continue",
                        color = if (hasRecorded) Color.Black else Color.Gray,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

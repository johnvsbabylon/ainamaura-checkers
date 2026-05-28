package ai.ainamaura.checkers.ui.main

import android.app.Application
import android.view.MotionEvent
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation3.runtime.NavKey
import ai.ainamaura.checkers.Move
import ai.ainamaura.checkers.Position
import ai.ainamaura.checkers.getValidMoves
import ai.ainamaura.checkers.theme.*

class MainScreenViewModelFactory(val application: Application) : androidx.lifecycle.ViewModelProvider.Factory {
    override fun <T : androidx.lifecycle.ViewModel> create(modelClass: Class<T>): T {
        return MainScreenViewModel(application) as T
    }
}

@Composable
fun MainScreen(
    onItemClick: (NavKey) -> Unit,
    modifier: Modifier = Modifier,
    sharedViewModel: MainScreenViewModel? = null
) {
    val application = LocalContext.current.applicationContext as Application
    val viewModel: MainScreenViewModel = sharedViewModel
        ?: viewModel(factory = MainScreenViewModelFactory(application))
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Background)
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title — gradient cursive
        Text(
            text = "ainamaura checkers",
            fontSize = 30.sp,
            fontWeight = FontWeight.Light,
            fontFamily = FontFamily.Cursive,
            textAlign = TextAlign.Center,
            style = TextStyle(
                brush = Brush.horizontalGradient(
                    listOf(NeonBlue, NeonPurple, NeonOrange, NeonTeal)
                )
            ),
            modifier = Modifier.padding(bottom = 6.dp)
        )

        // Mode buttons row
        ModeButtonsRow(uiState.mode, viewModel::setMode)

        // Status text
        Text(
            text = uiState.statusText,
            color = Color.LightGray,
            fontSize = 11.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )

        // Content area based on active tab
        when (uiState.activeTab) {
            AppTab.PLAY -> {
                CheckersBoardView(
                    uiState = uiState,
                    onMove = viewModel::handleMove
                )
                Spacer(modifier = Modifier.height(10.dp))
                // Compact chat below board
                ChatPanelView(
                    messages = uiState.chatMessages,
                    onSendMessage = viewModel::handleChatInput,
                    modifier = Modifier.weight(1f)
                )
            }
            AppTab.CHAT -> {
                ChatPanelView(
                    messages = uiState.chatMessages,
                    onSendMessage = viewModel::handleChatInput,
                    onSendImage = viewModel::handleImageSeed,
                    modifier = Modifier.weight(1f)
                )
            }
            AppTab.MODEL -> {
                NpuVisualizerView(
                    uiState = uiState,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        // Tab bar
        TabBar(activeTab = uiState.activeTab, onTabSelect = viewModel::setActiveTab)
    }
}

// ============================================================
// MODE BUTTONS
// ============================================================

@Composable
fun ModeButtonsRow(currentMode: String, onModeChange: (String) -> Unit) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.padding(vertical = 4.dp)
    ) {
        Button(
            onClick = { onModeChange("teach") },
            colors = ButtonDefaults.buttonColors(
                containerColor = if (currentMode == "teach") NeonTeal else SurfaceMid
            ),
            shape = RoundedCornerShape(8.dp)
        ) {
            Text(
                "Teach Me",
                color = if (currentMode == "teach") Color.Black else Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 13.sp
            )
        }
        Button(
            onClick = { onModeChange("beat") },
            colors = ButtonDefaults.buttonColors(
                containerColor = if (currentMode == "beat") NeonPurple else SurfaceMid
            ),
            shape = RoundedCornerShape(8.dp)
        ) {
            Text("Beat Me", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
        }
    }
}

// ============================================================
// TAB BAR
// ============================================================

@Composable
fun TabBar(activeTab: AppTab, onTabSelect: (AppTab) -> Unit) {
    Row(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
        listOf(AppTab.PLAY to "Play", AppTab.CHAT to "Chat", AppTab.MODEL to "NPU").forEach { (tab, label) ->
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clickable { onTabSelect(tab) }
                    .background(
                        if (activeTab == tab) NeonPurple.copy(alpha = 0.15f)
                        else Color.Transparent
                    )
                    .border(
                        BorderStroke(
                            1.dp,
                            if (activeTab == tab) NeonPurple.copy(alpha = 0.5f)
                            else BorderFaint
                        )
                    )
                    .padding(vertical = 10.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    label,
                    color = if (activeTab == tab) NeonPurple else Color.Gray,
                    fontSize = 12.sp,
                    fontWeight = if (activeTab == tab) FontWeight.Bold else FontWeight.Normal
                )
            }
        }
    }
}

// ============================================================
// CHECKERS BOARD
// ============================================================

@Composable
fun CheckersBoardView(
    uiState: UiState,
    onMove: (Move) -> Unit
) {
    var selectedPos by remember { mutableStateOf<Position?>(null) }
    val validMoves = remember(uiState.board) { getValidMoves(uiState.board, "human") }

    // Teach mode hint
    val teachHint: Move? = if (uiState.mode == "teach" && validMoves.isNotEmpty() && !uiState.isAiTurn) {
        // Pick best move for human as hint
        validMoves.maxByOrNull { it.captures.size }
    } else null

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(8.dp))
            .background(Color.Black)
            .border(1.dp, CardDarker, RoundedCornerShape(8.dp))
    ) {
        val tileSize = maxWidth / 8

        Canvas(modifier = Modifier
            .fillMaxSize()
            .pointerInput(uiState.board, uiState.isAiTurn) {
                detectTapGestures { offset ->
                    if (uiState.isAiTurn) return@detectTapGestures

                    val col = (offset.x / tileSize.toPx()).toInt().coerceIn(0, 7)
                    val row = (offset.y / tileSize.toPx()).toInt().coerceIn(0, 7)
                    val pos = Position(row, col)

                    if (selectedPos != null) {
                        val move = validMoves.find { it.from == selectedPos && it.to == pos }
                        if (move != null) {
                            onMove(move)
                            selectedPos = null
                        } else {
                            val cell = uiState.board[row][col]
                            if (cell != null && cell.player == "human") {
                                selectedPos = pos
                            } else {
                                selectedPos = null
                            }
                        }
                    } else {
                        val cell = uiState.board[row][col]
                        if (cell != null && cell.player == "human") {
                            selectedPos = pos
                        }
                    }
                }
            }) {

            val tileSizePx = tileSize.toPx()

            // Draw board squares
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val isDark = (r + c) % 2 == 1
                    val squareColor = if (isDark) GridDark else GridLight

                    drawRect(
                        color = squareColor,
                        topLeft = Offset(c * tileSizePx, r * tileSizePx),
                        size = androidx.compose.ui.geometry.Size(tileSizePx, tileSizePx)
                    )
                }
            }

            // Draw selected piece highlight
            if (selectedPos != null) {
                drawRect(
                    color = Color(0x3314B8A6), // teal 20% opacity
                    topLeft = Offset(selectedPos!!.col * tileSizePx, selectedPos!!.row * tileSizePx),
                    size = androidx.compose.ui.geometry.Size(tileSizePx, tileSizePx)
                )

                // Draw valid move target dots
                validMoves.filter { it.from == selectedPos }.forEach { move ->
                    drawCircle(
                        color = NeonPurple,
                        radius = 8.dp.toPx(),
                        center = Offset(
                            move.to.col * tileSizePx + tileSizePx / 2,
                            move.to.row * tileSizePx + tileSizePx / 2
                        )
                    )
                }
            }

            // Draw Teach Me hint borders
            if (teachHint != null && selectedPos == null) {
                // Source square: NeonTeal border
                val srcLeft = teachHint.from.col * tileSizePx
                val srcTop = teachHint.from.row * tileSizePx
                drawRect(
                    color = NeonTeal,
                    topLeft = Offset(srcLeft, srcTop),
                    size = androidx.compose.ui.geometry.Size(tileSizePx, tileSizePx),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5.dp.toPx())
                )
                // Destination square: NeonOrange border
                val dstLeft = teachHint.to.col * tileSizePx
                val dstTop = teachHint.to.row * tileSizePx
                drawRect(
                    color = NeonOrange,
                    topLeft = Offset(dstLeft, dstTop),
                    size = androidx.compose.ui.geometry.Size(tileSizePx, tileSizePx),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5.dp.toPx())
                )
            }

            // Draw pieces
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val piece = uiState.board[r][c] ?: continue

                    val center = Offset(
                        c * tileSizePx + tileSizePx / 2,
                        r * tileSizePx + tileSizePx / 2
                    )
                    val pieceRadius = tileSizePx * 0.42f

                    // Piece gradient
                    val pieceBrush = if (piece.player == "human") {
                        Brush.linearGradient(
                            listOf(NeonTeal, NeonOrange),
                            start = Offset(center.x - pieceRadius, center.y - pieceRadius),
                            end = Offset(center.x + pieceRadius, center.y + pieceRadius)
                        )
                    } else {
                        Brush.linearGradient(
                            listOf(NeonBlue, NeonPurple),
                            start = Offset(center.x - pieceRadius, center.y - pieceRadius),
                            end = Offset(center.x + pieceRadius, center.y + pieceRadius)
                        )
                    }

                    drawCircle(
                        brush = pieceBrush,
                        radius = pieceRadius,
                        center = center
                    )
                    // Border
                    drawCircle(
                        color = Color.White.copy(alpha = 0.2f),
                        radius = pieceRadius,
                        center = center,
                        style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.dp.toPx())
                    )

                    // King star
                    if (piece.isKing) {
                        // Draw ★ as a small white circle indicator since Canvas doesn't have drawText
                        drawCircle(
                            color = Color.White,
                            radius = pieceRadius * 0.3f,
                            center = center
                        )
                    }
                }
            }
        }
    }
}

// ============================================================
// CHAT PANEL
// ============================================================

@Composable
fun ChatPanelView(
    messages: List<String>,
    onSendMessage: (String) -> Unit,
    onSendImage: ((android.net.Uri) -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val imagePickerLauncher = if (onSendImage != null) {
        androidx.activity.compose.rememberLauncherForActivityResult(
            contract = androidx.activity.result.contract.ActivityResultContracts.GetContent()
        ) { uri: android.net.Uri? ->
            uri?.let { onSendImage(it) }
        }
    } else null

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardDark),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.dp, Border)
    ) {
        Column(modifier = Modifier.fillMaxSize().padding(8.dp)) {
            // Header
            Text(
                text = "Ainamaura Core Chat (Onboard Model)",
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
                color = NeonPurple,
                modifier = Modifier.padding(bottom = 6.dp)
            )

            // Messages
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                state = listState,
                reverseLayout = true,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                items(messages.reversed()) { message ->
                    val isHuman = message.startsWith("Human:")
                    ChatBubble(message = message, isHuman = isHuman)
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Input row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                BasicTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    textStyle = TextStyle(
                        color = Color.White,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace
                    ),
                    modifier = Modifier
                        .weight(1f)
                        .background(Color.Black, RoundedCornerShape(6.dp))
                        .border(1.dp, Border, RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 8.dp),
                    decorationBox = { innerTextField ->
                        if (inputText.isEmpty()) {
                            Text(
                                "Type a message...",
                                color = Color.Gray,
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                        innerTextField()
                    }
                )
                Button(
                    onClick = {
                        if (inputText.isNotBlank()) {
                            onSendMessage(inputText.trim())
                            inputText = ""
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = NeonPurple),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text("Send", fontSize = 11.sp, color = Color.White)
                }
                if (imagePickerLauncher != null) {
                    Button(
                        onClick = { imagePickerLauncher.launch("image/*") },
                        colors = ButtonDefaults.buttonColors(containerColor = NeonTeal),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Text("📷", fontSize = 14.sp)
                    }
                }
            }
        }
    }
}

@Composable
fun ChatBubble(message: String, isHuman: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isHuman) Arrangement.End else Arrangement.Start
    ) {
        Card(
            colors = CardDefaults.cardColors(
                containerColor = if (isHuman) SurfaceMid else CardDarker
            ),
            border = BorderStroke(
                1.dp,
                if (isHuman) NeonOrange.copy(0.2f) else NeonTeal.copy(0.2f)
            ),
            shape = RoundedCornerShape(8.dp),
            modifier = Modifier.widthIn(max = 280.dp)
        ) {
            Text(
                message,
                fontSize = 11.sp,
                color = Color.White,
                modifier = Modifier.padding(8.dp)
            )
        }
    }
}

// ============================================================
// NPU VISUALIZER — "The Crown Jewel"
// ============================================================

@Composable
fun NpuVisualizerView(
    uiState: UiState,
    modifier: Modifier = Modifier
) {
    val mambaState = uiState.mambaState
    val weights = mambaState.weights

    // Pulsing dot animation
    val infiniteTransition = rememberInfiniteTransition(label = "npu_pulse")
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_alpha"
    )

    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
    ) {
        // Card container with gradient top edge
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = CardDark.copy(alpha = 0.9f)),
            shape = RoundedCornerShape(12.dp),
            border = BorderStroke(1.dp, Border)
        ) {
            Column {
                // Gradient top line
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(2.dp)
                        .background(
                            Brush.horizontalGradient(
                                listOf(NeonBlue, NeonPurple, NeonOrange, NeonTeal)
                            )
                        )
                )

                Column(modifier = Modifier.padding(12.dp)) {
                    // Header row
                    NpuHeader(mambaState, pulseAlpha)

                    Spacer(modifier = Modifier.height(16.dp))

                    // Section 0: Live internal state visualization (Ainamaura shows you her mind)
                    uiState.stateVisualizationBitmap?.let { bmp ->
                        Text(
                            "Internal State",
                            fontSize = 10.sp,
                            color = Color.Gray,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.padding(bottom = 6.dp)
                        )
                        androidx.compose.foundation.Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "Ainamaura internal state visualization",
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(80.dp)
                                .clip(RoundedCornerShape(6.dp))
                                .border(1.dp, BorderFaint, RoundedCornerShape(6.dp))
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                    }

                    // Section 1: Mamba h Vector
                    MambaHVectorSection(mambaState)

                    Spacer(modifier = Modifier.height(16.dp))

                    // Section 2: FHN Neurons
                    FhnNeuronSection(mambaState, pulseAlpha)

                    Spacer(modifier = Modifier.height(16.dp))

                    // Section 3: Attention Heatmap
                    AttentionHeatmapSection(mambaState)
                }
            }
        }
    }
}

@Composable
fun NpuHeader(mambaState: ai.ainamaura.checkers.MambaState, pulseAlpha: Float) {
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            // Pulsing blue dot
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(NeonBlue.copy(alpha = pulseAlpha))
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                "NPU Neuromorphic Co-Processor",
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                fontFamily = FontFamily.Monospace
            )
        }
        Text(
            "Weightless Mamba SSM Core v3.0 • Continuous Training Active",
            fontSize = 9.sp,
            color = Color.Gray,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(start = 16.dp, top = 2.dp)
        )
        Spacer(modifier = Modifier.height(8.dp))

        // Stats row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            NpuBadge(
                dotColor = NeonTeal,
                text = "Neurons: ${formatNeuronCount(mambaState.virtualNeuronCount)} / 200M"
            )
            NpuBadge(
                dotColor = NeonPurple,
                text = "Cycles: ${mambaState.continuousTrainingCycles}"
            )
            Box(
                modifier = Modifier
                    .background(CardDarker, RoundedCornerShape(4.dp))
                    .border(1.dp, BorderFaint, RoundedCornerShape(4.dp))
                    .padding(horizontal = 6.dp, vertical = 3.dp)
            ) {
                Text(
                    "NPU CORES ACTIVE",
                    fontSize = 8.sp,
                    color = NeonTeal,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun NpuBadge(dotColor: Color, text: String) {
    Row(
        modifier = Modifier
            .background(CardDarker, RoundedCornerShape(4.dp))
            .border(1.dp, BorderFaint, RoundedCornerShape(4.dp))
            .padding(horizontal = 6.dp, vertical = 3.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(5.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(text, fontSize = 9.sp, color = Color.Gray, fontFamily = FontFamily.Monospace)
    }
}

// ============================================================
// SECTION 1: Mamba h Vector
// ============================================================

@Composable
fun MambaHVectorSection(mambaState: ai.ainamaura.checkers.MambaState) {
    Column {
        // Label row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                "Mamba SSM Vector h_t [16-dim]",
                fontSize = 10.sp,
                color = Color.Gray,
                fontFamily = FontFamily.Monospace
            )
            Text(
                "Δ_t = ${"%.4f".format(mambaState.delta)}",
                fontSize = 10.sp,
                color = NeonPurple,
                fontFamily = FontFamily.Monospace
            )
        }

        Spacer(modifier = Modifier.height(6.dp))

        // 8-column grid of cells (2 rows x 8 cols for 16 dims)
        for (rowStart in intArrayOf(0, 8)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(3.dp)
            ) {
                for (idx in rowStart until minOf(rowStart + 8, mambaState.h.size)) {
                    val value = mambaState.h[idx]
                    val pct = ((value + 1.2) * 0.45).coerceIn(0.0, 1.0).toFloat()

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(50.dp)
                            .background(CardDark, RoundedCornerShape(4.dp))
                            .border(1.dp, BorderFaint, RoundedCornerShape(4.dp))
                            .padding(4.dp)
                    ) {
                        // Fill bar from bottom
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .fillMaxHeight(pct)
                                .align(Alignment.BottomCenter)
                                .background(
                                    Brush.verticalGradient(
                                        listOf(
                                            NeonPurple.copy(0.1f),
                                            NeonBlue.copy(0.05f),
                                            Color.Transparent
                                        )
                                    )
                                )
                        )
                        Column(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                "h[$idx]",
                                fontSize = 7.sp,
                                color = Color.Gray,
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                "${"%.3f".format(value)}",
                                fontSize = 9.sp,
                                color = Color.White,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(3.dp))
        }

        // Equation label
        Text(
            "h_t = exp(Δ_t • A_bar) • h_t-1 + (Δ_t • B_bar) • x_t",
            fontSize = 8.sp,
            color = NeonBlue.copy(alpha = 0.6f),
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(top = 4.dp)
        )
    }
}

// ============================================================
// SECTION 2: FHN Neurons
// ============================================================

@Composable
fun FhnNeuronSection(mambaState: ai.ainamaura.checkers.MambaState, pulseAlpha: Float) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                "FitzHugh-Nagumo Spiking Neurons (Euler dt=0.15)",
                fontSize = 10.sp,
                color = Color.Gray,
                fontFamily = FontFamily.Monospace
            )
            Text(
                "Action Potentials",
                fontSize = 10.sp,
                color = NeonOrange,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.height(6.dp))

        // 10 neuron rows
        for (i in 0 until 10) {
            val v = mambaState.fhnVoltage[i]
            val w = mambaState.fhnRecovery[i]
            val isSpiked = v > 1.0

            val vPct = ((v + 1.5) / 3.0).coerceIn(0.0, 1.0).toFloat()
            val wPct = ((w + 1.0) / 2.0).coerceIn(0.0, 1.0).toFloat()

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(
                        if (isSpiked) Color(0x331A0800) else Color(0x0AFFFFFF)
                    )
                    .border(
                        1.dp,
                        if (isSpiked) NeonOrange.copy(0.8f) else BorderFaint,
                        RoundedCornerShape(8.dp)
                    )
                    .padding(6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Pulsing dot
                val dotScale = if (isSpiked) pulseAlpha * 0.25f + 1f else 1f
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .scale(dotScale)
                        .clip(CircleShape)
                        .background(
                            if (isSpiked) NeonOrange
                            else if (v > 0.5) Color(0xFFF59E0B)
                            else Color.DarkGray
                        )
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    "N_$i",
                    fontSize = 10.sp,
                    color = Color.Gray,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.width(28.dp)
                )

                // Bars column
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    // Voltage bar (6dp tall)
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(RoundedCornerShape(99.dp))
                            .background(CardDark)
                    ) {
                        Box(
                            Modifier
                                .fillMaxWidth(vPct)
                                .fillMaxHeight()
                                .clip(RoundedCornerShape(99.dp))
                                .background(
                                    if (isSpiked) Brush.horizontalGradient(
                                        listOf(NeonOrange, Color(0xFFEF4444))
                                    )
                                    else Brush.horizontalGradient(
                                        listOf(NeonBlue, NeonPurple)
                                    )
                                )
                        )
                    }
                    // Recovery bar (4dp tall)
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(4.dp)
                            .clip(RoundedCornerShape(99.dp))
                            .background(CardDark)
                    ) {
                        Box(
                            Modifier
                                .fillMaxWidth(wPct)
                                .fillMaxHeight()
                                .clip(RoundedCornerShape(99.dp))
                                .background(NeonTeal)
                        )
                    }
                }

                Spacer(Modifier.width(8.dp))
                Text(
                    "v:${"%.3f".format(v)}",
                    fontSize = 9.sp,
                    color = NeonBlue,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(Modifier.width(4.dp))
                Text(
                    "w:${"%.3f".format(w)}",
                    fontSize = 9.sp,
                    color = NeonTeal,
                    fontFamily = FontFamily.Monospace
                )
            }
            Spacer(modifier = Modifier.height(3.dp))
        }
    }
}

// ============================================================
// SECTION 3: Attention Heatmap
// ============================================================

@Composable
fun AttentionHeatmapSection(mambaState: ai.ainamaura.checkers.MambaState) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                "Transformer Multi-Head Attention Head Map",
                fontSize = 10.sp,
                color = Color.Gray,
                fontFamily = FontFamily.Monospace
            )
            Text(
                "Q x K^T / √d",
                fontSize = 10.sp,
                color = NeonTeal,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.height(6.dp))

        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Heatmap grid
            Box(
                modifier = Modifier
                    .size(140.dp)
                    .background(CardDark, RoundedCornerShape(8.dp))
                    .border(1.dp, BorderFaint, RoundedCornerShape(8.dp))
                    .padding(4.dp)
            ) {
                Column {
                    repeat(8) { row ->
                        Row(modifier = Modifier.weight(1f)) {
                            repeat(8) { col ->
                                val alpha = mambaState.attentionMatrix[row][col].toFloat().coerceIn(0f, 1f)
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .fillMaxHeight()
                                        .padding(1.dp)
                                        .clip(RoundedCornerShape(2.dp))
                                        .background(NeonTeal.copy(alpha = alpha))
                                )
                            }
                        }
                    }
                }
            }

            // Legend
            Column(
                verticalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.padding(top = 4.dp)
            ) {
                LegendItem(color = NeonTeal, label = "High Target Salience (>0.6)")
                LegendItem(color = NeonTeal.copy(alpha = 0.3f), label = "Diagnostic Focus (>0.3)")
                LegendItem(color = Color(0xFF1A1A1A), label = "Ambient Scans (<0.1)")

                Text(
                    "Active Head filters: Center controls, flank threat lines, king safety cells.",
                    fontSize = 8.sp,
                    color = Color.Gray,
                    modifier = Modifier.widthIn(max = 180.dp)
                )
            }
        }
    }
}

@Composable
fun LegendItem(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(color, RoundedCornerShape(2.dp))
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(label, fontSize = 8.sp, color = Color.Gray)
    }
}

// ============================================================
// WALKIE TALKIE BUTTON (kept for backward compat)
// ============================================================

@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun WalkieTalkieButton(
    isListening: Boolean,
    onStartListening: () -> Unit,
    onStopListening: () -> Unit
) {
    val gradient = Brush.radialGradient(
        colors = listOf(
            if (isListening) NeonOrange else NeonPurple,
            Background
        )
    )

    Box(
        modifier = Modifier
            .size(80.dp)
            .clip(CircleShape)
            .background(gradient)
            .pointerInteropFilter { event ->
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        onStartListening()
                        true
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        onStopListening()
                        true
                    }
                    else -> false
                }
            },
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = if (isListening) "Listening..." else "Hold to Speak",
            color = Color.White,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

// ============================================================
// HELPERS
// ============================================================

private fun formatNeuronCount(count: Int): String {
    return when {
        count >= 1_000_000 -> "${"%.1f".format(count / 1_000_000.0)}M"
        count >= 1_000 -> "${"%.0f".format(count / 1_000.0)}K"
        else -> count.toString()
    }
}

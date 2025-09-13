#!/usr/bin/env python3
"""
Generate detailed academic sections for the 4 CNN models following the baseline structure.
Each section includes: Data Collection, Network Architecture, Training Objective, Summary, and Architecture Table.
"""

def generate_hybrid_cnn_section():
    """Generate detailed section for Hybrid CNN + RSSI"""
    
    section = """
\\subsubsection{Hybrid Convolutional Neural Network with RSSI Integration}

\\subsubsection{Data Collection}
The hybrid approach extends the baseline data collection framework by incorporating heterogeneous sensor modalities to enhance localization accuracy. In addition to the 52-subcarrier CSI fingerprints collected at each reference point, we extracted RSSI measurements from the same WiFi transmissions. Each training instance was represented as a composite feature vector $\\mathbf{X} = [\\mathbf{X}_{CSI}, \\mathbf{X}_{RSSI}]$, where $\\mathbf{X}_{CSI} \\in \\mathbb{R}^{52 \\times 1}$ contained amplitude-only CSI data and $\\mathbf{X}_{RSSI} \\in \\mathbb{R}^{1}$ represented the received signal strength indicator. This multi-modal representation yielded supervised training pairs of the form:
\\begin{equation}\\label{eq:hybrid_training}
\\big([\\mathbf{X}_{CSI}, \\mathbf{X}_{RSSI}], (x,y)\\big)
\\end{equation}
where $(x,y) \\in \\mathbb{R}^2$ denotes the ground-truth coordinates.

\\subsubsection{Network Architecture}
The hybrid architecture implements a dual-branch neural network designed to exploit the complementary strengths of CSI and RSSI measurements. The CSI branch processes fine-grained spectral information to capture environment-specific multipath signatures, while the RSSI branch provides coarse-grained distance estimates that constrain the spatial search space.

The CSI processing branch employs a multi-scale convolutional structure with two parallel paths. The first path utilizes 32 filters with kernel size 3 to capture local spectral variations, such as frequency-selective fading notches and sharp phase transitions. The second path applies 32 filters with kernel size 7 to extract broader spectral patterns that characterize the overall channel response. Both paths incorporate batch normalization and max pooling (pool size 2) to ensure training stability and reduce dimensionality. The outputs are concatenated and processed through an additional convolutional layer with 64 filters (kernel size 3) to learn higher-order feature interactions. Global average pooling compresses the resulting feature maps, and a dense layer with 128 units produces the final CSI feature representation.

The RSSI processing branch consists of a dedicated three-layer fully connected network. The input RSSI value is processed through dense layers with 32, 32, and 32 units, respectively, each employing ReLU activations. This design allows the network to learn nonlinear transformations of the RSSI signal while maintaining computational efficiency.

Feature fusion occurs through concatenation of the 128-dimensional CSI features and 32-dimensional RSSI features, yielding a 160-dimensional joint representation. This fused feature vector is processed by three dense layers with 256, 128, and 64 units, incorporating dropout (rates 0.3, 0.2) to prevent overfitting. The final linear output layer produces the predicted coordinates $(\\hat{x}, \\hat{y})$.

\\subsubsection{Training Objective}
Training employed the same Euclidean distance loss function as the baseline:
\\[
L = \\sqrt{(x - \\hat{x})^2 + (y - \\hat{y})^2}
\\]
However, the multi-modal nature of the input required careful consideration of feature scaling and fusion strategies. The network was trained end-to-end, allowing gradient-based optimization to learn optimal feature combination weights automatically. This approach ensured that the complementary information from CSI (fine-grained spatial signatures) and RSSI (coarse distance estimates) was effectively integrated for improved localization accuracy.

\\subsubsection{Summary}
The hybrid CNN + RSSI model demonstrates that multi-modal sensor fusion can significantly enhance indoor localization performance. By processing CSI and RSSI data through specialized neural pathways and fusing their outputs, the model achieves superior accuracy compared to single-modality approaches. The architecture's ability to leverage both fine-grained spectral features and coarse-grained signal strength information makes it particularly robust to environmental variations and measurement noise.

\\renewcommand{\\arraystretch}{1.3}
\\begin{table}[htbp]
\\centering
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{|c|l|c|c|c|c|}
\\hline
\\textbf{Layer} & \\textbf{Type} & \\textbf{Filter / Kernel} & \\textbf{Activation} & \\textbf{Output Shape} & \\textbf{Parameters} \\\\ \\hline
0 & CSI Input & -- & -- & (52, 1) & 0 \\\\ \\hline
1 & RSSI Input & -- & -- & (1) & 0 \\\\ \\hline
2 & Conv1D (Path 1) & 32 filters, kernel=3, padding=same & ReLU & (52, 32) & 128 \\\\ \\hline
3 & Conv1D (Path 2) & 32 filters, kernel=7, padding=same & ReLU & (52, 32) & 256 \\\\ \\hline
4 & BatchNormalization & -- & -- & (26, 32) each & 128 \\\\ \\hline
5 & MaxPooling1D & pool size=2 & -- & (26, 32) each & 0 \\\\ \\hline
6 & Concatenate & -- & -- & (26, 64) & 0 \\\\ \\hline
7 & Conv1D & 64 filters, kernel=3, padding=same & ReLU & (26, 64) & 12,352 \\\\ \\hline
8 & GlobalAveragePooling1D & -- & -- & (64) & 0 \\\\ \\hline
9 & Dense (CSI) & 128 units & ReLU & (128) & 8,320 \\\\ \\hline
10 & Dense (RSSI) & 32 units & ReLU & (32) & 64 \\\\ \\hline
11 & Dense (RSSI) & 32 units & ReLU & (32) & 1,056 \\\\ \\hline
12 & Dense (RSSI) & 32 units & ReLU & (32) & 1,056 \\\\ \\hline
13 & Concatenate & -- & -- & (160) & 0 \\\\ \\hline
14 & Dense & 256 units & ReLU & (256) & 41,216 \\\\ \\hline
15 & Dropout & rate=0.3 & -- & (256) & 0 \\\\ \\hline
16 & Dense & 128 units & ReLU & (128) & 32,896 \\\\ \\hline
17 & Dropout & rate=0.2 & -- & (128) & 0 \\\\ \\hline
18 & Dense & 64 units & ReLU & (64) & 8,256 \\\\ \\hline
19 & Dense & 2 units & Linear & (2) & 130 \\\\ \\hline
\\end{tabular}%
}
\\caption{Layer configuration of the hybrid CNN + RSSI model}
\\label{tab:hybrid_cnn}
\\end{table}
"""
    return section

def generate_attention_cnn_section():
    """Generate detailed section for Attention CNN"""
    
    section = """
\\subsubsection{Attention-based Convolutional Neural Network}

\\subsubsection{Data Collection}
The attention-based approach utilizes the same CSI amplitude data collection protocol as previous models, with each training instance represented as $\\mathbf{X} \\in \\mathbb{R}^{52 \\times 1}$ containing amplitude information from all 52 subcarriers. However, the key innovation lies in the network's ability to dynamically weight the importance of different frequency components during the localization process. Training pairs maintain the form:
\\begin{equation}\\label{eq:attention_training}
\\big(\\mathbf{X}, (x,y)\\big)
\\end{equation}
where $\\mathbf{X}$ represents the amplitude-only CSI fingerprint and $(x,y)$ denotes the spatial coordinates.

\\subsubsection{Network Architecture}
The attention mechanism addresses a fundamental limitation of traditional CNNs: the assumption that all subcarriers contribute equally to localization accuracy. In reality, different frequency components exhibit varying sensitivity to spatial positioning due to frequency-selective fading and multipath propagation characteristics.

The architecture begins with two sequential convolutional layers, each employing 64 filters. The first layer uses a kernel size of 5 to capture medium-range spectral dependencies, while the second layer applies a kernel size of 3 for fine-grained pattern extraction. Both layers incorporate batch normalization to stabilize training dynamics and feature distributions.

The core innovation lies in the self-attention mechanism, which implements learnable query, key, and value transformations. Given the feature representation $\\mathbf{H} \\in \\mathbb{R}^{52 \\times 64}$ from the convolutional layers, we compute:
\\begin{align}
\\mathbf{Q} &= \\mathbf{H}\\mathbf{W}_Q, \\quad \\mathbf{K} = \\mathbf{H}\\mathbf{W}_K, \\quad \\mathbf{V} = \\mathbf{H}\\mathbf{W}_V \\\\
\\mathbf{A} &= \\text{softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right) \\\\
\\mathbf{Z} &= \\mathbf{A}\\mathbf{V}
\\end{align}
where $\\mathbf{W}_Q, \\mathbf{W}_K, \\mathbf{W}_V \\in \\mathbb{R}^{64 \\times 64}$ are learnable projection matrices, and $d_k = 64$ provides appropriate scaling for the attention scores.

The attention output $\\mathbf{Z}$ is combined with the original features through a residual connection and layer normalization:
\\[
\\mathbf{H}' = \\text{LayerNorm}(\\mathbf{H} + \\mathbf{Z})
\\]
This design preserves important spectral information while allowing the network to emphasize the most discriminative frequency components for each spatial location.

Following attention processing, global average pooling compresses the enhanced feature representation, and three dense layers (256, 128 units with dropout rates 0.3, 0.2) perform the final coordinate regression.

\\subsubsection{Training Objective}
The attention model employs the standard Euclidean distance loss:
\\[
L = \\sqrt{(x - \\hat{x})^2 + (y - \\hat{y})^2}
\\]
The attention mechanism is trained end-to-end through backpropagation, learning to identify which subcarriers provide the most reliable spatial information for different regions of the environment. This adaptive weighting allows the model to automatically discover frequency-dependent localization patterns that may not be apparent through traditional signal processing approaches.

\\subsubsection{Summary}
The attention-based CNN represents a significant advancement in learning-based localization by incorporating adaptive frequency weighting. The self-attention mechanism enables the network to discover which subcarriers are most informative for spatial positioning, leading to more robust and accurate localization performance. This approach is particularly valuable in complex multipath environments where frequency-selective fading creates spatially-dependent spectral signatures.

\\renewcommand{\\arraystretch}{1.3}
\\begin{table}[htbp]
\\centering
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{|c|l|c|c|c|c|}
\\hline
\\textbf{Layer} & \\textbf{Type} & \\textbf{Filter / Kernel} & \\textbf{Activation} & \\textbf{Output Shape} & \\textbf{Parameters} \\\\ \\hline
0 & Input & -- & -- & (52, 1) & 0 \\\\ \\hline
1 & Conv1D & 64 filters, kernel=5, padding=same & ReLU & (52, 64) & 384 \\\\ \\hline
2 & BatchNormalization & -- & -- & (52, 64) & 256 \\\\ \\hline
3 & Conv1D & 64 filters, kernel=3, padding=same & ReLU & (52, 64) & 12,352 \\\\ \\hline
4 & BatchNormalization & -- & -- & (52, 64) & 256 \\\\ \\hline
5 & Dense (Query) & 64 units & Linear & (52, 64) & 4,160 \\\\ \\hline
6 & Dense (Key) & 64 units & Linear & (52, 64) & 4,160 \\\\ \\hline
7 & Dense (Value) & 64 units & Linear & (52, 64) & 4,160 \\\\ \\hline
8 & Dot Product & Q·K^T/√64 & -- & (52, 52) & 0 \\\\ \\hline
9 & Softmax & -- & Softmax & (52, 52) & 0 \\\\ \\hline
10 & Dot Product & Attention·V & -- & (52, 64) & 0 \\\\ \\hline
11 & Add & Residual connection & -- & (52, 64) & 0 \\\\ \\hline
12 & LayerNormalization & -- & -- & (52, 64) & 128 \\\\ \\hline
13 & GlobalAveragePooling1D & -- & -- & (64) & 0 \\\\ \\hline
14 & Dense & 256 units & ReLU & (256) & 16,640 \\\\ \\hline
15 & Dropout & rate=0.3 & -- & (256) & 0 \\\\ \\hline
16 & Dense & 128 units & ReLU & (128) & 32,896 \\\\ \\hline
17 & Dropout & rate=0.2 & -- & (128) & 0 \\\\ \\hline
18 & Dense & 2 units & Linear & (2) & 258 \\\\ \\hline
\\end{tabular}%
}
\\caption{Layer configuration of the attention-based CNN}
\\label{tab:attention_cnn}
\\end{table}
"""
    return section

def generate_multiscale_cnn_section():
    """Generate detailed section for Multi-Scale CNN"""
    
    section = """
\\subsubsection{Multi-Scale Convolutional Neural Network}

\\subsubsection{Data Collection}
The multi-scale approach employs the standard amplitude-only CSI data collection framework, with each training instance represented as $\\mathbf{X} \\in \\mathbb{R}^{52 \\times 1}$. The key insight driving this architecture is that wireless channel characteristics manifest at multiple frequency scales simultaneously: local fading effects create sharp spectral variations, while global propagation patterns produce broader frequency responses. Training pairs follow the established format:
\\begin{equation}\\label{eq:multiscale_training}
\\big(\\mathbf{X}, (x,y)\\big)
\\end{equation}
where the network learns to extract and combine features across different spectral scales for enhanced localization accuracy.

\\subsubsection{Network Architecture}
The multi-scale design addresses the limitation of fixed receptive fields in traditional CNNs by implementing parallel processing paths with varying kernel sizes. This approach enables simultaneous capture of local spectral irregularities and global channel characteristics that occur at different frequency scales.

Three parallel convolutional paths process the input CSI data simultaneously. Path 1 employs 32 filters with kernel size 3 to capture local spectral features such as frequency-selective fading notches and rapid phase variations across adjacent subcarriers. Path 2 utilizes 32 filters with kernel size 7 to extract medium-scale patterns that characterize cluster-based multipath propagation. Path 3 implements 32 filters with kernel size 15 to capture global spectral envelope characteristics that reflect the overall channel response and room geometry.

Each path incorporates batch normalization to ensure stable training dynamics across different scales, followed by max pooling (pool size 2) to reduce computational complexity while preserving essential pattern information. The three parallel outputs are concatenated along the feature dimension, yielding a 96-dimensional representation that encompasses spectral patterns at multiple resolutions.

An additional convolutional layer with 128 filters (kernel size 3) processes the concatenated multi-scale features to learn higher-order interactions between different frequency scales. This integration step is crucial for combining complementary information from local, medium, and global spectral patterns. Global average pooling compresses the resulting feature maps while preserving spatial invariance properties.

The final processing stages consist of three dense layers with 256, 128 units, incorporating dropout (rates 0.3, 0.2) to prevent overfitting. This hierarchical structure enables the network to progressively refine the multi-scale spectral representation into precise coordinate predictions.

\\subsubsection{Training Objective}
Training utilizes the Euclidean distance loss function:
\\[
L = \\sqrt{(x - \\hat{x})^2 + (y - \\hat{y})^2}
\\]
The multi-scale architecture requires careful gradient flow management across parallel paths to ensure balanced learning of features at different scales. The end-to-end training approach allows the network to automatically determine optimal weighting between local, medium, and global spectral features based on their discriminative power for spatial localization.

\\subsubsection{Summary}
The multi-scale CNN demonstrates that explicit modeling of frequency-domain hierarchies significantly enhances localization performance. By processing CSI data at multiple spectral resolutions simultaneously, the network captures complementary information that spans from fine-grained fading patterns to broad channel characteristics. This comprehensive spectral analysis enables more robust localization in complex multipath environments where spatial signatures manifest across different frequency scales.

\\renewcommand{\\arraystretch}{1.3}
\\begin{table}[htbp]
\\centering
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{|c|l|c|c|c|c|}
\\hline
\\textbf{Layer} & \\textbf{Type} & \\textbf{Filter / Kernel} & \\textbf{Activation} & \\textbf{Output Shape} & \\textbf{Parameters} \\\\ \\hline
0 & Input & -- & -- & (52, 1) & 0 \\\\ \\hline
1 & Conv1D (Path 1) & 32 filters, kernel=3, padding=same & ReLU & (52, 32) & 128 \\\\ \\hline
2 & Conv1D (Path 2) & 32 filters, kernel=7, padding=same & ReLU & (52, 32) & 256 \\\\ \\hline
3 & Conv1D (Path 3) & 32 filters, kernel=15, padding=same & ReLU & (52, 32) & 512 \\\\ \\hline
4 & BatchNormalization & -- & -- & (52, 32) each & 384 \\\\ \\hline
5 & MaxPooling1D & pool size=2 & -- & (26, 32) each & 0 \\\\ \\hline
6 & Concatenate & -- & -- & (26, 96) & 0 \\\\ \\hline
7 & Conv1D & 128 filters, kernel=3, padding=same & ReLU & (26, 128) & 36,992 \\\\ \\hline
8 & BatchNormalization & -- & -- & (26, 128) & 512 \\\\ \\hline
9 & GlobalAveragePooling1D & -- & -- & (128) & 0 \\\\ \\hline
10 & Dropout & rate=0.3 & -- & (128) & 0 \\\\ \\hline
11 & Dense & 256 units & ReLU & (256) & 33,024 \\\\ \\hline
12 & Dropout & rate=0.3 & -- & (256) & 0 \\\\ \\hline
13 & Dense & 128 units & ReLU & (128) & 32,896 \\\\ \\hline
14 & Dropout & rate=0.2 & -- & (128) & 0 \\\\ \\hline
15 & Dense & 2 units & Linear & (2) & 258 \\\\ \\hline
\\end{tabular}%
}
\\caption{Layer configuration of the multi-scale CNN}
\\label{tab:multiscale_cnn}
\\end{table}
"""
    return section

def generate_residual_cnn_section():
    """Generate detailed section for Residual CNN"""
    
    section = """
\\subsubsection{Residual Convolutional Neural Network}

\\subsubsection{Data Collection}
The residual architecture utilizes amplitude-only CSI data following the established collection protocol, with training instances represented as $\\mathbf{X} \\in \\mathbb{R}^{52 \\times 1}$. The motivation for this approach stems from the observation that deeper networks can potentially learn more complex spatial-spectral relationships, but suffer from vanishing gradient problems during training. The residual framework addresses this limitation while maintaining the standard training pair format:
\\begin{equation}\\label{eq:residual_training}
\\big(\\mathbf{X}, (x,y)\\big)
\\end{equation}
where the network learns hierarchical feature representations through progressively deeper processing stages.

\\subsubsection{Network Architecture}
The residual CNN implements skip connections inspired by ResNet architectures to enable effective training of deeper networks while preserving gradient flow. The core innovation lies in the residual blocks, which allow the network to learn incremental feature refinements rather than complete transformations, leading to more stable training dynamics and improved feature reuse.

The architecture begins with an initial convolutional layer employing 32 filters with kernel size 7 to establish a broad receptive field for capturing global spectral patterns. This layer is followed by batch normalization to ensure stable feature distributions throughout the network.

Three residual blocks form the core of the architecture, each implementing the fundamental residual learning principle. A residual block consists of two convolutional layers with batch normalization, where the input $\\mathbf{x}$ is added to the output $\\mathbf{F}(\\mathbf{x})$ to form the block output $\\mathbf{y} = \\mathbf{F}(\\mathbf{x}) + \\mathbf{x}$. This formulation allows the network to learn residual mappings $\\mathbf{F}(\\mathbf{x}) = \\mathbf{y} - \\mathbf{x}$, which are often easier to optimize than direct mappings.

The first residual block processes the initial features with 32 filters (kernel size 3), maintaining spatial resolution while learning local spectral patterns. Max pooling (pool size 2) follows to reduce dimensionality. The second residual block increases the channel depth to 64 filters, enabling more complex feature interactions while applying another pooling operation. The third residual block further expands to 128 filters, capturing high-level abstractions of the spectral-spatial relationships.

When the number of input and output channels differs between residual connections, a 1×1 convolutional projection layer adjusts the dimensionality to enable proper skip connections. This ensures that gradient information can flow directly from deeper layers to earlier ones without degradation.

Global average pooling compresses the final feature maps while preserving translation invariance. Three dense layers (256, 128 units with dropout rates 0.3, 0.2) perform the final coordinate regression, benefiting from the rich hierarchical features learned through the residual blocks.

\\subsubsection{Training Objective}
The residual model employs the Euclidean distance loss:
\\[
L = \\sqrt{(x - \\hat{x})^2 + (y - \\hat{y})^2}
\\]
The skip connections facilitate gradient flow during backpropagation, enabling effective training of the deeper architecture. This improved optimization allows the network to learn more sophisticated spatial-spectral mappings while avoiding the vanishing gradient problem that typically limits the depth of traditional CNNs.

\\subsubsection{Summary}
The residual CNN demonstrates that network depth can be effectively leveraged for indoor localization when appropriate architectural innovations are employed. The skip connections enable training of deeper models that learn hierarchical feature representations spanning from low-level spectral patterns to high-level spatial abstractions. This approach achieves improved localization accuracy while maintaining training stability through better gradient flow and feature reuse mechanisms.

\\renewcommand{\\arraystretch}{1.3}
\\begin{table}[htbp]
\\centering
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{|c|l|c|c|c|c|}
\\hline
\\textbf{Layer} & \\textbf{Type} & \\textbf{Filter / Kernel} & \\textbf{Activation} & \\textbf{Output Shape} & \\textbf{Parameters} \\\\ \\hline
0 & Input & -- & -- & (52, 1) & 0 \\\\ \\hline
1 & Conv1D & 32 filters, kernel=7, padding=same & ReLU & (52, 32) & 256 \\\\ \\hline
2 & BatchNormalization & -- & -- & (52, 32) & 128 \\\\ \\hline
3 & ResBlock1-Conv1D & 32 filters, kernel=3, padding=same & ReLU & (52, 32) & 3,104 \\\\ \\hline
4 & ResBlock1-BatchNorm & -- & -- & (52, 32) & 128 \\\\ \\hline
5 & ResBlock1-Conv1D & 32 filters, kernel=3, padding=same & Linear & (52, 32) & 3,104 \\\\ \\hline
6 & ResBlock1-BatchNorm & -- & -- & (52, 32) & 128 \\\\ \\hline
7 & ResBlock1-Add & Skip connection & ReLU & (52, 32) & 0 \\\\ \\hline
8 & MaxPooling1D & pool size=2 & -- & (26, 32) & 0 \\\\ \\hline
9 & ResBlock2-Conv1D & 64 filters, kernel=3, padding=same & ReLU & (26, 64) & 6,208 \\\\ \\hline
10 & ResBlock2-BatchNorm & -- & -- & (26, 64) & 256 \\\\ \\hline
11 & ResBlock2-Conv1D & 64 filters, kernel=3, padding=same & Linear & (26, 64) & 12,352 \\\\ \\hline
12 & ResBlock2-BatchNorm & -- & -- & (26, 64) & 256 \\\\ \\hline
13 & ResBlock2-Projection & 64 filters, kernel=1 & Linear & (26, 64) & 2,112 \\\\ \\hline
14 & ResBlock2-Add & Skip connection & ReLU & (26, 64) & 0 \\\\ \\hline
15 & MaxPooling1D & pool size=2 & -- & (13, 64) & 0 \\\\ \\hline
16 & ResBlock3-Conv1D & 128 filters, kernel=3, padding=same & ReLU & (13, 128) & 24,704 \\\\ \\hline
17 & ResBlock3-BatchNorm & -- & -- & (13, 128) & 512 \\\\ \\hline
18 & ResBlock3-Conv1D & 128 filters, kernel=3, padding=same & Linear & (13, 128) & 49,280 \\\\ \\hline
19 & ResBlock3-BatchNorm & -- & -- & (13, 128) & 512 \\\\ \\hline
20 & ResBlock3-Projection & 128 filters, kernel=1 & Linear & (13, 128) & 8,320 \\\\ \\hline
21 & ResBlock3-Add & Skip connection & ReLU & (13, 128) & 0 \\\\ \\hline
22 & GlobalAveragePooling1D & -- & -- & (128) & 0 \\\\ \\hline
23 & Dense & 256 units & ReLU & (256) & 33,024 \\\\ \\hline
24 & Dropout & rate=0.3 & -- & (256) & 0 \\\\ \\hline
25 & Dense & 128 units & ReLU & (128) & 32,896 \\\\ \\hline
26 & Dropout & rate=0.2 & -- & (128) & 0 \\\\ \\hline
27 & Dense & 2 units & Linear & (2) & 258 \\\\ \\hline
\\end{tabular}%
}
\\caption{Layer configuration of the residual CNN}
\\label{tab:residual_cnn}
\\end{table}
"""
    return section

def main():
    """Generate all four model sections"""
    
    print("🚀 GENERATING DETAILED ACADEMIC SECTIONS FOR 4 CNN MODELS")
    print("="*70)
    
    # Generate sections
    hybrid_section = generate_hybrid_cnn_section()
    attention_section = generate_attention_cnn_section() 
    multiscale_section = generate_multiscale_cnn_section()
    residual_section = generate_residual_cnn_section()
    
    # Combine all sections
    full_document = f"""
% Four CNN Model Detailed Sections
% Following the structure of the baseline CNN section

{hybrid_section}

{attention_section}

{multiscale_section}

{residual_section}
"""
    
    # Save to file
    with open("four_cnn_model_sections.tex", "w", encoding="utf-8") as f:
        f.write(full_document)
    
    print("✅ Generated detailed sections for 4 CNN models")
    print("📄 Saved to: four_cnn_model_sections.tex")
    print("\n📋 SECTIONS INCLUDED:")
    print("1. Hybrid CNN + RSSI (Multi-modal Fusion)")
    print("2. Attention CNN (Adaptive Learning)")  
    print("3. Multi-Scale CNN (Multi-Scale Processing)")
    print("4. Residual CNN (Deep Learning)")
    print("\n📝 Each section includes:")
    print("   • Data Collection methodology")
    print("   • Network Architecture details")
    print("   • Training Objective explanation") 
    print("   • Summary and insights")
    print("   • Complete architecture table")

if __name__ == "__main__":
    main()



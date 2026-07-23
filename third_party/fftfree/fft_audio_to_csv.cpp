// fft_audio_to_csv.cpp
// Reads a WAV file, runs FFT using repo code, outputs real, imag, mag as CSV
#include "eigen_fft.hpp" // Or your preferred FFT header
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

// Minimal WAV reader (mono, 16-bit PCM)
std::vector<float> read_wav(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open WAV file");
    file.seekg(44); // Skip header
    std::vector<float> data;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        data.push_back(static_cast<float>(sample));
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: fft_audio_to_csv input.wav output.csv\n";
        return 1;
    }
    std::string wavfile = argv[1];
    std::string csvfile = argv[2];
    auto samples = read_wav(wavfile);
    size_t N = samples.size();

    // FFT using repo code
    std::vector<std::complex<float>> fft_out(N);
    fft_forward(samples.data(), fft_out.data(), N); // Replace with your FFT function

    std::ofstream out(csvfile);
    out << "real,imag,mag\n";
    for (size_t i = 0; i < N; ++i) {
        float real = fft_out[i].real();
        float imag = fft_out[i].imag();
        float mag = std::abs(fft_out[i]);
        out << real << "," << imag << "," << mag << "\n";
    }
    std::cout << "Wrote FFT CSV to " << csvfile << std::endl;
    return 0;
}

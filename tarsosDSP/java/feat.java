import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteOrder;
import java.util.Arrays;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.TarsosDSPAudioFormat;
import be.tarsos.dsp.io.UniversalAudioInputStream;
import be.tarsos.dsp.mfcc.MFCC;

public class feat {

	public static void main(String args[]) {

		String inputFileDir[] = { "wav" };
		String outputFileDir[] = { "feat" };

		final int SAMPLING_RATE = 16000;
		final int FRAME_SIZE = 640; // 40 msec
		final int FRAME_OVERLAP = 480; // 30 msec
		final int FRAME_MOVE = FRAME_SIZE - FRAME_OVERLAP;
		final int FEATURE_PER_FRAME = 40; // 40 Frame -> 1 feature input
		final int NUM_MEL_FILTER_BANK = 40; // 40 dimension Filter bank
		final int MFCC_PER_FRAME = 40; // only for mfcc
		final float LOWER_FILTER_FREQ = 64.0f;
		final float UPPER_FILTER_FREQ = 8000.0f;

		MFCC mfcc = new MFCC(FRAME_SIZE, SAMPLING_RATE, MFCC_PER_FRAME, NUM_MEL_FILTER_BANK, LOWER_FILTER_FREQ,
				UPPER_FILTER_FREQ);
		TarsosDSPAudioFormat format = new TarsosDSPAudioFormat(TarsosDSPAudioFormat.Encoding.PCM_SIGNED, SAMPLING_RATE,
				16, 1, 2, SAMPLING_RATE, ByteOrder.BIG_ENDIAN.equals(ByteOrder.nativeOrder()));
		
		for (int i = 0; i < inputFileDir.length; i++) {
			File in = new File(inputFileDir[i]);
			File out = new File(inputFileDir[i]);

			File input;
			File output;

			File[] fileList = in.listFiles();

			for (File tFile : fileList) {
				String inputName = inputFileDir[i] + "/" + tFile.getName();
				String outputName = outputFileDir[i] + "/" + tFile.getName().replace(".wav", ".fb40txt");

				input = new File(inputName);
				output = new File(outputName);

				try {
					System.out.println(inputName);
					FileInputStream audioStream = new FileInputStream(input);
					BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(output));

					AudioDispatcher dispatcher = new AudioDispatcher(new UniversalAudioInputStream(audioStream, format),
							FRAME_SIZE, FRAME_OVERLAP);

				
					dispatcher.addAudioProcessor(new AudioProcessor() {
						
						int cnt = 0;

						@Override
						public void processingFinished() {
							try {
								bufferedWriter.close();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}

						@Override
						public boolean process(AudioEvent audioEvent) {
						
							float[] audioFloatBuffer = audioEvent.getFloatBuffer().clone();
							float bin[] = mfcc.magnitudeSpectrum(audioFloatBuffer);
							
							int [] centerFreq = mfcc.getCenterFrequencies().clone();
			                centerFreq[NUM_MEL_FILTER_BANK] = (centerFreq[NUM_MEL_FILTER_BANK+1] + centerFreq[NUM_MEL_FILTER_BANK-1]) / 2;
			                
							float fbank[] = mfcc.melFilter(bin, centerFreq);
							cnt++;
						
							if (cnt != 1) {
								try {
									bufferedWriter.write(Arrays.toString(fbank).replace("[", "").replace("]", "")); bufferedWriter.newLine();
									
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
							return true;
						}
					});

					dispatcher.run();

				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}


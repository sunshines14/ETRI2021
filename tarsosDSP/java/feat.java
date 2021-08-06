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

import java.lang.Math; 
public class feat {

	public static void main(String args[]) {

		String inputFileDir[] = { "wav" };
		String outputFileDir[] = { "feat" };

		final int SAMPLING_RATE = 16000;
		
		final int FRAME_SIZE = 640; // 40 msec
		final int FRAME_OVERLAP = 493; // for 40time bins
		//final int FRAME_OVERLAP = 320; 
		final int FRAME_MOVE = FRAME_SIZE - FRAME_OVERLAP;
		final int FEATURE_PER_FRAME = 40; // 40 Frame -> 1 feature input
		final int NUM_MEL_FILTER_BANK = 40; // 40 dimension Filter bank
		/*
		final int SAMPLING_RATE = 44100;
		final int FRAME_SIZE = 2072; // 47 msec
                final int FRAME_OVERLAP = 1036; // 23 msec
                final int FRAME_MOVE = FRAME_SIZE - FRAME_OVERLAP;
                final int FEATURE_PER_FRAME = 16; // 40 Frame -> 1 feature input
                final int NUM_MEL_FILTER_BANK = 128; // 40 dimension Filter bank
		*/
		final int MFCC_PER_FRAME = 40; // only for mfcc
		final float LOWER_FILTER_FREQ = 0.0f;
		final float UPPER_FILTER_FREQ = 8000;
		
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
							//System.out.println(cnt + Arrays.toString(audioFloatBuffer));
							//System.out.println("AudioDataFloat1 : Length: " + FRAME_MOVE + " Data: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, 0, FRAME_MOVE)));
							//System.out.println("AudioDataFloat2 : Length: " + FRAME_MOVE + " Data: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, FRAME_MOVE, FRAME_MOVE * 2)));
							//System.out.println("AudioDataFloat3 : Length: " + FRAME_MOVE + " Data: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, FRAME_MOVE * 2, FRAME_MOVE * 3)));
							//System.out.println("AudioDataFloat4 : Length: " + FRAME_MOVE + " Data: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, FRAME_MOVE * 3, FRAME_MOVE * 4)));
							float bin[] = mfcc.magnitudeSpectrum(audioFloatBuffer);
							int [] centerFrequencies = new int[40 + 2];

        						centerFrequencies[0] = Math.round(0 / 16000 * 640);
        						centerFrequencies[centerFrequencies.length - 1] = (int) (640 / 2);

       				 			double mel[] = new double[2];
        						mel[0] = (float) (2595 * (float) (Math.log(1 + 0.0f/700) / Math.log(10)));
        						mel[1] = (float) (2595 * (float) (Math.log(1 + 8000.0f/700) / Math.log(10)));
        
        						float factor = (float)((mel[1] - mel[0]) / (40 + 1));
        						//Calculates te centerfrequencies.
        						for (int i = 1; i <= 40; i++) {
            							float fc = ((float) (700 * (Math.pow(10, (mel[0] + factor * i) / 2595) -1)) / 16000) * 640;
            							centerFrequencies[i] = Math.round(fc);
							}

							float fbank[] = mfcc.melFilter(bin, centerFrequencies);
							
							for (int i = 0; i < 40; i++) {
								fbank[i] = (float)Math.log(fbank[i]);
							}
							
							//float f[] = mfcc.nonLinearTransformation(fbank).clone();
							//float m[] = mfcc.cepCoefficients(f);
							cnt++;	
							if (cnt != 1) {
								try {
									//System.out.println(cnt + Arrays.toString(fbank));
									//bufferedWriter.write("Frame 1: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, 0, FRAME_MOVE))); bufferedWriter.newLine();
									//bufferedWriter.write("Frame 2: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, FRAME_MOVE, FRAME_MOVE * 2))); bufferedWriter.newLine();
									//bufferedWriter.write("Frame 3: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, FRAME_MOVE * 2, FRAME_MOVE * 3))); bufferedWriter.newLine();
									//bufferedWriter.write("Frame 4: " + Arrays.toString(Arrays.copyOfRange(audioFloatBuffer, FRAME_MOVE * 3, FRAME_MOVE * 4))); bufferedWriter.newLine();
									
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

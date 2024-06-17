package ch03_.operator.sec04_accuracy;

public class AccuracyExample3 {
	public static void main(String[] args) {
		int apple = 1;
		double pieceUnit = 0.1;
		int number = 7;
		
		double comp = number * (pieceUnit * 10.0); 
		System.out.println("comp: " + comp);
		
		double result = ((apple * 10) - comp) / 10.0;
		
		
		System.out.println("사과 1개에서 남은 양: " + result);
		
		/* comp: 7.0
		사과 1개에서 남은 양: 0.3 */
	}
}

package ch03_.operator.sec04_accuracy;

public class AccuracyExample1 {
	public static void main(String[] args) {
		int apple = 1;
		double pieceUnit = 0.1;
		int number = 7;
		
		double result = apple - number*pieceUnit; // 오차 발생 : 기대 0.3
		System.out.println("사과 1개에서 남은 양: " + result);
		// 사과 1개에서 남은 양: 0.29999999999999993
	}
}

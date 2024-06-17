package ch04_.statement.sec02_conditions;

public class IfElseExample {
	public static void main(String[] args) {
		int score = 85;

		if(score>=90) { // 참인 경우
			System.out.println("점수가 90보다 큽니다.");
			System.out.println("등급은 A입니다.");
		} 
		else { // 옵션 : 거짓인 경우
			System.out.println("점수가 90보다 작습니다.");
			System.out.println("등급은 B입니다.");
		}
	}
}
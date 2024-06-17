package ch04_.statement.sec02_conditions;

public class IfElseIfElseExample2 {
	public static void main(String[] args) {
		int score = 80;
		
		if(score<70) {
			System.out.println("점수가 70 미만입니다.");
			System.out.println("등급은 D입니다.");
		} else if(score<80) { // python에서는 elif
			System.out.println("점수가 70~79입니다.");
			System.out.println("등급은 C입니다.");
		} else if(score<90) {
			System.out.println("점수가 80~89입니다.");
			System.out.println("등급은 B입니다.");
		} else { // 90보다 크면
			System.out.println("점수는 90~100입니다.");
			System.out.println("등급은 A입니다.");
		}
	}	
}


//작은 값 ->큰 값
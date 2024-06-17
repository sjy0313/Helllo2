package ch04_.statement.sec02_conditions;

public class IfNestedExample {
	public static void main(String[] args) {
		int score = (int)(Math.random()*20) + 81; // 0~0.1 * 20 -> 0 이상 20미만 난수 생성 + 81
		// 0 부터 100까지 난수 생성 
		System.out.println("점수: " + score);
		
		String grade;
		
		if(score>=90) { // 90~100
			if(score>=95) {
				grade = "A+";
			} else {
				grade = "A";
			}
		} else { // 81~89
			if(score>=85) {
				grade = "B+";
			} else {
				grade = "B";
			}
		}
		
		System.out.println("학점: " + grade);
	}
}
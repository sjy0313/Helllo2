package ch03_.operator.sec11_conditional;
/*
 * 삼항 조건 연산자 
 * python 참고 :YSIT24/Python/Syntax/Stmt/s01-if-4.py
 */
public class ConditionalOperationExample2 {
	public static void main(String[] args) {
		int score = 70;
		
		char grade = (score >= 90) ? 'A' : 
					 (score >= 80) ? 'B' :
					 (score >= 80) ? 'C' : 'D';
		
		System.out.printf("점수(%d), 등급(%c)\n", score, grade); // 점수(70), 등급(D)
		
	}
}


package ch04_.statement.sec02_conditions;
/*
 * 조건식이 만족하면(true이면) 불록 안의 실행문을 처리한다

 * if(조건식) {
 * 		실행문
 */
public class IfExample {
	public static void main(String[] args) {
		// int score = 93;
		int score = 89;
		// if 다음 (조건식)안은 boolean 자료형으로 출력되지만 
		// 파이썬과 같이 ()를 제거하고 조건식을 작성하면 오류발생됨. 
		
		if(score >= 90) {
			System.out.println("점수가 90보다 큽니다.");
			System.out.println("등급은 A입니다.");
		}
		
		// 블록의 시작({)과 종료(}) 기호를 생략가능 
		// 생략하면 조건식의 다음에 나오는 한 문장만 블록으로 처리한다 
		
		if(score < 90)
			System.out.println("점수가 90보다 작습니다.");
		
		System.out.println("등급은 B입니다.");
	}
}
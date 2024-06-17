package ch04_.statement.sec03_switches;
/*
 * [switch expressions]
 * java12 이후부터 지원 
 * break문을 없앰
 */
public class SwitchExpressionsExample {
	public static void main(String[] args) {
		char grade = 'B';
		
		switch(grade) {
			case 'A', 'a' -> {
				System.out.println("우수 회원입니다.");
			}
			case 'B', 'b' -> {
				System.out.println("일반 회원입니다.");
			}						
			default -> {
				System.out.println("손님입니다.");
			}
		}
		// [문제] if문으로 변경
		switch(grade) {
			case 'A', 'a' -> System.out.println("우수 회원입니다.");
			case 'B', 'b' -> System.out.println("일반 회원입니다.");						
			default -> System.out.println("손님입니다.");
		}
	}
}
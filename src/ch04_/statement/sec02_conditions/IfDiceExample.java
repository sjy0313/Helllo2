package ch04_.statement.sec02_conditions;

public class IfDiceExample {
	public static void main(String[] args) {
		// Math.random()은 0.0 <= 임의의 실수 생성 < 1.0 
		//(int) casting -> 실수 -> 정수 -> 소수점을 버림. 
		// 0부터 5까지 정수에서 + 1 을 통해 1 부터 6까지 정수 생성 	
		int num = (int)(Math.random()*6) + 1;
		
		if(num==1) {
			System.out.println("1번이 나왔습니다.");
		} else if(num==2) {	
			System.out.println("2번이 나왔습니다.");
		} else if(num==3) {
			System.out.println("3번이 나왔습니다.");
		} else if(num==4) {
			System.out.println("4번이 나왔습니다.");
		} else if(num==5) {
			System.out.println("5번이 나왔습니다.");
		} else {
			System.out.println("6번이 나왔습니다.");
		}
	}
}
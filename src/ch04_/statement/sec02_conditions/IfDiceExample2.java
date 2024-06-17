package ch04_.statement.sec02_conditions;
/*
 * 주사위 : 랜덤(random)함수 이용
 * Math.random() : 0.0 < 1.0
 * 값의 변화 알아보기 
 * 
 */
public class IfDiceExample2 {
	public static void main(String[] args) {
		// Math.random()은 0.0 <= 임의의 실수 생성 < 1.0 
		//(int) casting -> 실수 -> 정수 -> 소수점을 버림. 
		// 0부터 5까지 정수에서 + 1 을 통해 1 부터 6까지 정수 생성 	
		double rnd = Math.random();
		double rnd6 = rnd * 6;
		int rndx = (int)rnd6;
		int num = rndx + 1;
		System.out.printf("난수(%f), 경우수(%f), 정수(%d), 결과(%d)\n", rnd, rnd6, rndx, num);
		
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
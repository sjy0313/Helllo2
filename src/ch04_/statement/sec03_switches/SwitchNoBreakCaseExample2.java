package ch04_.statement.sec03_switches;

public class SwitchNoBreakCaseExample2 {
	public static void main(String[] args) {
		int time = (int)(Math.random()*4) + 8;  
		System.out.println("[현재시간: " + time + " 시]");
		/*
		 * case, default 의 위치는 관계가 없다
		 * 관례 : 
		 * 	1. case의 값은 작은 값에서 큰 값 순으로 배치
		 *  2. default는 맨 마지막에 배치
		 */
		switch(time) {
		default:
			System.out.println("외근을 나갑니다.");
			case 8:
				System.out.println("출근합니다.");
			case 9:
				System.out.println("회의를 합니다.");
			case 10:
				System.out.println("업무를 봅니다.");
		}
	}
}



package ch04_.statement.sec04.forloop;

public class PrintFrom1To10Example5 {
	public static void main(String[] args) {
	/*
	 * 반복문을 사용하여 1부터 10까지의 1씩 증가하는 연속된 수를 생성하여 총 합:
	 * 단, 진행되는 상황을 출력하라
	 */
		int sum = 0;
		for(int i = 1; i <= 10; i++) {
			sum += i;
			System.out.println("현재숫자 :" + i + "현재 합 : " + sum);
			
		}
		
		System.out.println("총 합: " + sum);
	}
}

   
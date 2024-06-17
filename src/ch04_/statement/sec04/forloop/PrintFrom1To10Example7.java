package ch04_.statement.sec04.forloop;

public class PrintFrom1To10Example7 {
	public static void main(String[] args) {
	/*
	 * 반복문을 사용하여 1부터 10까지의 1씩 증가하는 연속된 수를 생성하여 짝수의 합을 구해라
	 * 
	 */
		int sum = 0;
		
		for(int i = 1; i <= 10; i++) {
			if(i % 2 == 0) { // 조건문 추가 )뒤에 ;제거 후 {로 묶어주기
				sum += i;
				System.out.printf("i=%d, sum=%d\n", i, sum);	
			}
		}
		
		System.out.printf("1부터 10까지의 연속된 수에서 짝수의 합은? %d\n", sum);
	}
}






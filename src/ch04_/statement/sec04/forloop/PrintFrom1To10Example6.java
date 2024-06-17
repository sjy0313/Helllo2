package ch04_.statement.sec04.forloop;

public class PrintFrom1To10Example6 {
	public static void main(String[] args) {
	/*
	 * 반복문을 사용하여 1부터 10까지의 1씩 증가하는 연속된 수를 생성하여 총 합:
	 * 단, 진행되는 상황을 출력하라
	 */
		int sum = 0;
		for(int i = 1; i <= 10; i++) {
			sum += i;
			System.out.printf("i=%d, sum=%d\n", i, sum);	
		}
		
		System.out.printf("1부터 10까지 합은? %d\n", sum);
	}
}

   // for()문 안에서 s=0 을 선언해주면 s cannot be resolved to a variable error발생
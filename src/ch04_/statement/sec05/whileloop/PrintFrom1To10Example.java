package ch04_.statement.sec05.whileloop;

public class PrintFrom1To10Example {
	public static void main(String[] args) {
		int i = 1; 
		// 초기식과 증감식 생략
		while (/*초기식*/; i<=10; /*초기식*/) { // 조건식이 참이면 블록을 실행(반복)
			System.out.print(i + " ");
			i++;
		}
	}
}

package ch04_.statement.sec04.forloop;

public class PrintFrom1To10Example2 {
	public static void main(String[] args) {
		
		// 무한루프 : 무한반복, 종료x
		// 조건식(i <= 10)을 만족하는 동안 계속 반복
		for(int i=1; i<=10;) { // 1,2,3,4 ... 10  i를 증가
			System.out.println(i + " "); // println -> 1 무한출력
		}
	}
}
package ch04_.statement.sec04.forloop;

public class PrintFrom1To10Example4 {
	public static void main(String[] args) {
		
		int i = 0;
		// 조건식을 기술하지 않으면 무한루프 
		// i값이 무한히 증가 
		for(;;) { // 1,2,3,4 ... 10  i를 증가
			System.out.println(i++); 
		}
	}
}
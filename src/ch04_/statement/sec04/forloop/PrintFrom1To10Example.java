package ch04_.statement.sec04.forloop;

public class PrintFrom1To10Example {
	public static void main(String[] args) {
		
	
		for(int i=1; i<=10; i++) { // 1,2,3,4 ... 10  i를 증가
			int x = i * i;
			//System.out.printf("i=%d, x=%d\n", i, x); 
			System.out.printf("루프에서 선언된 변수 i는?", + i); 
		}
	}
}

	// 루프나 블록 안에서 선언된 변수는 밖에서 참조할 수 없다
	// 루프나 블록을 벗어나면 선언된 변수는 사라짐
	
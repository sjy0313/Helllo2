package ch06_.classes.sec08.methods.exam02.params;

public class Computer {
	//가변길이 매개변수를 갖는 메소드 선언
	// 인자 : 다중인자/ 배열처리
	// int sums_array(int ... values) { // ... : 가변길이(배열은 아니지만 배열변수처럼 사용)
	   int sum(int ... values) {
		//sum 변수 선언
		int sum = 0;
		
		//values 는 배열 타입의 변수처럼 사용
		for (int i = 0; i < values.length; i++) {
			sum += values[i];
		}

		//합산 결과를 리턴
		return sum;
		
	}
	// 인자 : 다중인자(처리하지 못함), 배열
	   int sum(int[] values) { 
	   System.out.println("sum(int[])");
	   int sum = 0;
		
		for (int i = 0; i < values.length; i++) {
			sum += values[i];
		}

		//합산 결과를 리턴
		return sum;
		
	}
	
	
	
	
}
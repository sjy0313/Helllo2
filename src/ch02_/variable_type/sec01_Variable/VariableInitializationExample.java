package ch02_.variable_type.sec01_Variable;

public class VariableInitializationExample {
	public static void main(String[] args) {
		
		
		//변수 value 선언
		// 로컬변수 : 메소드 안에서 선언된 변수 
		int value;    
		// value 값이 결정되지 않았기 떄문에 오류발생
		//'The local variable value may not have been initialized'//
		
		// 로컬변수는 사용되기 전에 반드시 초기화 되어야 한다
		//연산 결과를 변수 result 의 초기값으로 대입
		int result = value + 10;
		
		//변수 result 값을 읽고 콘솔에 출력
		//System.out.println(result);
	}
}


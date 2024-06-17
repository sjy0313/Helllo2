package ch02_.variable_type.sec11_VariableScope;

public class VariableScopeExample {
	public static void main(String[] args) {
		int v1 = 15;
		if(v1>10) {
			// block 안에서 선언된 로컬 변수는 불록 안에서만 유효
			int v2 = v1 - 10;
			System.out.printf("[v2]" + v2);  // Unresolved compilation problem: 
			//v2 cannot be resolved to a variable (컴파일error)
			// 파일이 만들어지기는 하지만 오류발생
			/*
			 * cmd 창에서 확인(파일 d-drive temp example.java로 script저장 후 D: -> cd temp ->
			 * javac example.java cmd창에 입력. 
			 * 		example.java:10: error: cannot find symbol
            int v3 = v1 + v2 + 5;
                          ^
symbol:   variable v2
location: class example
1 error

			 */
			
		}
		//v2 변수를 사용할 수 없기 때문에 컴파일 에러 발생
		int v3 = v1 + v2 + 5; 
	}
}
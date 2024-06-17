package ch02_.variable_type.sec11_VariableScope;

public class VariableScopeExample2 {
	public static void main(String[] args) {
		int v1 = 15; // 매소드 블록에서 선언
		{ 
			// block 안에서 선언된 로컬 변수는 불록 안에서만 유효
			int v2 = v1 - 10;
			System.out.printf("[v2]" + v2); 
		}
			
		//v2 변수를 사용할 수 없기 때문에 컴파일 에러 발생
		// int v3 = v1 + v2 + 5; // v2 cannot be resolved to a variable
	}
}
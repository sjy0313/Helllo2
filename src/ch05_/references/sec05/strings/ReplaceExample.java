
package ch05_.references.sec05.strings;

public class ReplaceExample {
	public static void main(String[] args) {
		// 자바의 문자열의 요소는 참조하거나 바꿀 수 없다(불변)
		String oldStr = "자바 문자열은 불변입니다. 자바 문자열은 String입니다.";
		String newStr = oldStr.replace("자바", "JAVA");

		System.out.println(oldStr);
		System.out.println(newStr);
		
		System.out.println(oldStr == newStr); // python에서는 슬라이싱을 통한 참조가 가능했음. 
	}
}

/*python : 
 * original_string = "Hello, world!"
 * new_string = original_string.replace("world", "Python")
print(new_string)  # 출력: Hello, Python!
java :
String(문자열) 새로운 변수 = 기존변수.replace("world", "Python")
System.out.println(newStr);
*/

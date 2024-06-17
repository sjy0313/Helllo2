package ch02_.variable_type.sec06_String;

public class TextBlockExample {
	public static void main(String[] args) {
		String str1 = "" +
		"{\n" +
		"\t\"id\":\"winter\",\n" +
		"\t\"name\":\"눈송이\"\n" +
		"}";
		// 자바 13 버전부터 지원됨. 
		String str2 = """
				
		{
			"id":"winter",
			"name":"눈송이"
		}
		""";

		System.out.println(str1);
		System.out.println("------------------------------------");
		System.out.println(str2);
		System.out.println("------------------------------------");
		
		// 문자열 연결(\) 자바 14부터 지원 
		String str = """
		나는 자바를 \ 
		학습합니다.
		나는 자바 고수가 될 겁니다.
		""";
		System.out.println(str);
	}
}